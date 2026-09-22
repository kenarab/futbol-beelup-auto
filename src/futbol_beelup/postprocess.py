"""Ball detection and a smoothed virtual camera, with streaming final scaling."""
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


def fingerprint(video):
    digest = hashlib.sha256()
    with Path(video).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_ball(candidates, previous, elapsed, speed=1.5):
    """Candidates are normalized x/y/confidence. Reject implausible jumps."""
    if previous is not None:
        candidates = [item for item in candidates
                      if math.dist(item[:2], previous[:2]) <= speed * elapsed + .03]
    return max(candidates, key=lambda item: item[2], default=None)


def track(args):
    from .cli import probe, write_json
    if not Path(args.weights).is_file():
        raise ValueError('Supply a local detector weights file; no automatic model downloads.')
    if args.start < 0 or args.duration <= 0 or not 0 < args.sample_fps <= 60 or not 0 < args.confidence <= 1:
        raise ValueError('Invalid tracking range, sample rate or confidence.')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Tracking output already exists; choose another output path.')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise ValueError('Tracking needs optional dependencies: python -m pip install -e ".[gpu]" in your virtualenv.') from error
    metadata = probe(args.video)
    duration = min(args.duration, float(metadata['format']['duration']) - args.start)
    if duration <= 0:
        raise ValueError('Start is outside the video.')
    model = YOLO(args.weights)
    classes = [key for key, name in model.names.items() if name == args.ball_class]
    if not classes:
        raise ValueError(f'Model has no class named {args.ball_class!r}. Available: {model.names}')
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError('Cannot open video or determine frame rate.')
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    rows, previous, last_seen, next_sample = [], None, -1e9, 0.
    frame_index = 0
    try:
        while frame_index / fps < duration:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-6 < next_sample:
                continue
            next_sample = timestamp + 1 / args.sample_fps
            result = model.predict(frame, classes=classes, conf=args.confidence,
                                   imgsz=args.imgsz, device=args.device, verbose=False, save=False)[0]
            candidates = []
            if result.boxes is not None:
                for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    candidates.append(((x1+x2)/2/width, (y1+y2)/2/height, score))
            if timestamp - last_seen > 1:
                previous = None
            ball = choose_ball(candidates, previous, timestamp-last_seen)
            if ball is not None:
                previous, last_seen = ball, timestamp
            rows.append(dict(t=timestamp, ball=list(ball) if ball else None))
    finally:
        cap.release()
    if frame_index / fps < duration - max(.5, 2/fps):
        raise ValueError('Video decoding stopped before the requested tracking range ended.')
    document = dict(video=str(Path(args.video).resolve()), sha256=fingerprint(args.video),
                    start=args.start, duration=duration, width=width, height=height,
                    model=str(Path(args.weights).resolve()), sample_fps=args.sample_fps,
                    samples=rows, note='Experimental detections; null means ball not detected.')
    write_json(output, document)
    found = sum(row['ball'] is not None for row in rows)
    print(f'Saved {output}: ball detected in {found}/{len(rows)} sampled frames. Review before following.')


def camera_path(samples, duration, fps=25, hold=.8, response=.5, max_speed=.35):
    """Hold briefly during occlusion, then slowly return to center. Never invent ball positions."""
    previous_time = -1
    for row in samples:
        timestamp, ball = row['t'], row['ball']
        if not math.isfinite(timestamp) or not 0 <= timestamp < duration or timestamp <= previous_time:
            raise ValueError('Tracking timestamps must increase within the clip.')
        if ball is not None and (len(ball) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in ball)):
            raise ValueError('Invalid normalized ball detection.')
        previous_time = timestamp
    center, target, last_seen, index = [.5, .5], [.5, .5], -1e9, 0
    for frame in range(math.ceil(duration * fps)):
        timestamp = frame / fps
        while index < len(samples) and samples[index]['t'] <= timestamp:
            row = samples[index]
            if row['ball'] is not None:
                target = row['ball'][:2]
                last_seen = row['t']
            index += 1
        if timestamp - last_seen > hold:
            target = [.5, .5]
        for axis in (0, 1):
            delta = target[axis] - center[axis]
            if abs(delta) > .025:  # dead zone reduces jitter
                step = delta * (1 - math.exp(-1 / fps / response))
                center[axis] += max(-max_speed/fps, min(max_speed/fps, step))
        yield timestamp, *center


def follow(args):
    from .cli import probe, write_json
    document = json.loads(Path(args.manifest).read_text())
    video = Path(args.video or document['video']).resolve()
    if fingerprint(video) != document['sha256']:
        raise ValueError('Tracking belongs to a different video. Track the exact input to be rendered.')
    metadata = probe(video)
    stream = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    width, height = stream['width'], stream['height']
    if (width, height) != (document['width'], document['height']):
        raise ValueError('Tracking dimensions do not match source.')
    duration, start = document['duration'], document['start']
    if not 0 < duration or start < 0 or start+duration > float(metadata['format']['duration'])+.1:
        raise ValueError('Tracking range is outside the video.')
    if not .2 <= args.crop <= 1 or not 1 <= args.fps <= 60:
        raise ValueError('Use crop between .2 and 1, and FPS between 1 and 60.')
    # 16:9 crop contained in the source; resizing preserves this aspect ratio.
    crop_width = int(min(width * args.crop, height * 16 / 9) // 32) * 32
    crop_height = crop_width * 9 // 16
    if crop_width < 32:
        raise ValueError('Crop is too small.')
    out_width, out_height = {'native': (crop_width, crop_height), '720p': (1280,720), '1080p': (1920,1080)}[args.resolution]
    output = Path(args.output).resolve()
    if output.exists() or output == video:
        raise ValueError('Choose a new output file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        folder = Path(temporary)
        commands = folder / 'camera.txt'
        with commands.open('w') as handle:
            for timestamp, x, y in camera_path(document['samples'], duration, args.fps):
                left = round(max(0, min(width-crop_width, x*width-crop_width/2)))
                top = round(max(0, min(height-crop_height, y*height-crop_height/2)))
                handle.write(f'{timestamp:.6f} crop@ball x {left}, crop@ball y {top};\n')
        filters = (f'setpts=PTS-STARTPTS,fps={args.fps},sendcmd=f=camera.txt,'
                   f'crop@ball={crop_width}:{crop_height},'
                   f'scale={out_width}:{out_height}:flags=lanczos,setsar=1')
        # Relative command filename avoids filter escaping problems on external/Windows paths.
        partial = folder / 'render.mp4'
        subprocess.run(['ffmpeg','-v','error','-n','-ss',str(start),'-i',str(video),
                        '-t',str(duration),'-map','0:v:0','-map','0:a:0?', '-vf',filters,
                        '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                        '-c:a','aac','-movflags','+faststart',str(partial)], cwd=folder, check=True)
        partial.replace(output)
    write_json(output.with_suffix('.json'), dict(source=str(video), tracking=str(Path(args.manifest).resolve()),
               duration=duration, resolution=[out_width,out_height], crop=[crop_width,crop_height],
               upscaling='Lanczos interpolation; no recovered detail', missing_ball='hold then recenter'))
    print(f'Saved {output}')


def add_commands(commands):
    command = commands.add_parser('track', help='Detect ball positions using local YOLO weights')
    command.add_argument('video')
    command.add_argument('--weights', required=True)
    command.add_argument('--ball-class', default='sports ball')
    command.add_argument('--device', default='cpu', help='0 for the first NVIDIA GPU; cpu for CPU')
    command.add_argument('--start', type=float, default=0)
    command.add_argument('--duration', type=float, default=30)
    command.add_argument('--sample-fps', type=float, default=8)
    command.add_argument('--confidence', type=float, default=.35)
    command.add_argument('--imgsz', type=int, default=1280)
    command.add_argument('--output', default='outputs/ball-track.json')
    command.set_defaults(function=track)
    command = commands.add_parser('follow', help='Render smooth ball-following crop and optional final upscale')
    command.add_argument('manifest')
    command.add_argument('--video')
    command.add_argument('--crop', type=float, default=.8, help='Fraction of source width retained')
    command.add_argument('--fps', type=int, default=25)
    command.add_argument('--resolution', choices=['native','720p','1080p'], default='native')
    command.add_argument('--output', default='outputs/ball-follow.mp4')
    command.set_defaults(function=follow)
