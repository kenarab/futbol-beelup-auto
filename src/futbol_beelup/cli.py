#!/usr/bin/env python3
"""Download Beelup matches, rank windows with local Ollama, render highlights."""
import argparse
import base64
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from .storage import input_path, output_path, prepare_storage


def run(*args, capture=False):
    if args[0] == 'yt-dlp':
        args = (sys.executable, '-m', 'yt_dlp', *args[1:])
    return subprocess.run([str(a) for a in args], check=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def probe(path):
    return json.loads(run('ffprobe', '-v', 'error', '-show_format', '-show_streams',
                          '-of', 'json', path, capture=True))


def playlist_url(url, camera=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError('Use an HTTP(S) video URL.')
    if parsed.hostname in ('beelup.com', 'www.beelup.com') and parsed.path == '/player.php':
        query = urllib.parse.parse_qs(parsed.query)
        match_id = query.get('id', [''])[0]
        if not match_id.isdigit():
            raise ValueError('Beelup URL must contain a numeric id.')
        params = dict(id=match_id, camara=camera if camera is not None else query.get('c', [''])[0],
                      tipo='todo', formato='m3u8', desde='', duracion='')
        return 'https://beelup.com/obtener.video.playlist.php?' + urllib.parse.urlencode(params)
    return url


def download(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Destination: {directory.resolve()} ({shutil.disk_usage(directory).free / 2**30:.1f} GiB free)')
    source = playlist_url(args.url, args.camera)
    identity = directory / 'source.json'
    if identity.exists() and json.loads(identity.read_text())['playlist'] != source:
        raise ValueError('This directory belongs to another URL/camera. Choose a new directory.')
    write_json(identity, dict(url=args.url, playlist=source))
    request = urllib.request.Request(source, headers={'Referer': args.url, 'User-Agent': 'Mozilla/5.0'})
    expected = None
    if 'formato=m3u8' in source or urllib.parse.urlparse(source).path.endswith('.m3u8'):
        with urllib.request.urlopen(request, timeout=60) as response:
            manifest = response.read().decode()
        if not manifest.lstrip().startswith('#EXTM3U'):
            raise ValueError('No video playlist returned; the link may have expired.')
        (directory / 'source.m3u8').write_text(manifest)
        durations = [float(line.split(':', 1)[1].split(',')[0])
                     for line in manifest.splitlines() if line.startswith('#EXTINF:')]
        if durations:
            expected = sum(durations)
        if urllib.parse.urlparse(source).hostname == 'beelup.com':
            archive_originals(manifest, directory)
            originals = directory / 'originals'
            listing = originals / 'concat.txt'
            listing.write_text(''.join(f"file '{index:03d}.mp4'\n" for index in range(len(original_urls(manifest)))))
            video = directory / 'match.mp4'
            if not video.exists():
                partial = directory / 'match.partial.mp4'
                run('ffmpeg', '-v', 'warning', '-y', '-f', 'concat', '-safe', '1', '-i', listing,
                    '-map', '0', '-c', 'copy', '-bsf:v', 'filter_units=remove_types=12',
                    '-movflags', '+faststart', partial)
                partial.replace(video)
        else:
            video = None
    else:
        video = None
    if video is None:
        run('yt-dlp', '--no-skip-unavailable-fragments', '--fragment-retries', '10',
            '--retries', '5', '--concurrent-fragments', '8', '--referer', args.url,
            '--remux-video', 'mp4', '-o', directory / 'full-match.%(ext)s', source)
        video = directory / 'full-match.mp4'
    metadata = probe(video)
    actual = float(metadata['format']['duration'])
    if expected is not None and abs(actual - expected) > max(3, expected * .001):
        raise ValueError(f'Duration mismatch: expected {expected:.2f}s, got {actual:.2f}s.')
    write_json(directory / 'download.json', dict(url=args.url, playlist=source,
               expected_duration=expected, actual_duration=actual, media=metadata))
    print(f'Saved {video}: {actual / 60:.2f} minutes')


def original_urls(manifest):
    # This is the same conversion used by Beelup's player for source MP4s.
    return list(dict.fromkeys(line.replace('/hls/', '/').split('.mp4/')[0] + '.mp4'
                for line in manifest.splitlines()
                if line.startswith('https://') and '/hls/' in line and '.mp4/' in line))


def archive_originals(manifest, directory):
    urls = original_urls(manifest)
    if not urls:
        raise ValueError('Could not find original camera files in the Beelup playlist.')
    folder = directory / 'originals'
    folder.mkdir(exist_ok=True)
    write_json(directory / 'original-urls.json', urls)
    records = []
    for index, url in enumerate(urls):
        target = folder / f'{index:03d}.mp4'
        if not target.exists():
            partial = target.with_suffix('.mp4.part')
            run('curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
                '--max-time', '600', '--output', partial, url)
            partial.replace(target)
        metadata = probe(target)
        records.append(dict(file=target.name, url=url, media=metadata))
        print(f'Archived original {index + 1}/{len(urls)}', flush=True)
    write_json(directory / 'originals.json', records)


SCHEMA = {'type': 'object', 'properties': {
    'score': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'event': {'type': 'string', 'enum': ['chance', 'possible_goal', 'save', 'ordinary_play', 'unclear']},
    'reason': {'type': 'string'},
}, 'required': ['score', 'event', 'reason'], 'additionalProperties': False}
PROMPT = '''Review these chronological frames from one short amateur football window.
Rank how useful the window is for a reel of scoring chances (0 to 1).
Look for visible shots, saves, goalmouth action or a possible goal. Do not invent
ball trajectories, goals, team names or scores. If the ball/action cannot be seen,
use unclear and a low score. Ordinary possession is low value. Give a short reason
describing visible evidence and uncertainty. Return only the requested JSON.'''


def validate_result(result):
    score = result.get('score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Model returned an invalid score.')
    if result.get('event') not in SCHEMA['properties']['event']['enum'] or not isinstance(result.get('reason'), str):
        raise ValueError('Model returned an invalid event/reason.')
    return {key: result[key] for key in ('score', 'event', 'reason')}


def select_clips(windows, duration, top, minimum, padding):
    selected = []
    for window in sorted(windows, key=lambda item: item['score'], reverse=True):
        if window['score'] < minimum or window['event'] in ('unclear', 'ordinary_play'):
            continue
        candidate = dict(window, start=max(0, window['start'] - padding),
                         end=min(duration, window['end'] + padding))
        if any(candidate['start'] < item['end'] and candidate['end'] > item['start'] for item in selected):
            continue
        selected.append(candidate)
        if len(selected) == top:
            break
    return sorted(selected, key=lambda item: item['start'])


def analyze(args):
    if args.window <= 0 or args.frames < 2 or args.limit is not None and args.limit <= 0:
        raise ValueError('Use a positive window/limit and at least two frames.')
    if args.top <= 0 or not 0 <= args.minimum <= 1 or args.padding < 0:
        raise ValueError('Invalid selection settings.')
    host = urllib.parse.urlparse(args.ollama)
    if host.scheme != 'http' or host.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Run analysis on the GPU box using its local Ollama server.')
    video = Path(args.video).resolve()
    duration = float(probe(video)['format']['duration'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with video.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    settings = dict(sha256=digest.hexdigest(), model=args.model, window=args.window,
                    frames=args.frames, prompt=PROMPT, version=1)
    cache = output / 'windows.json'
    saved = json.loads(cache.read_text()) if cache.exists() else dict(settings=settings, windows=[])
    if saved['settings'] != settings:
        raise ValueError('Output contains analysis with different settings/video; use a new output directory.')
    count = math.ceil(duration / args.window)
    if args.limit is not None:
        count = min(count, args.limit)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for index in range(len(saved['windows']), count):
            start, end = index * args.window, min(duration, (index + 1) * args.window)
            images = []
            for frame in range(args.frames):
                timestamp = start + (end - start) * (frame + .5) / args.frames
                path = Path(temporary) / f'{frame}.jpg'
                run('ffmpeg', '-v', 'error', '-y', '-ss', timestamp, '-i', video,
                    '-frames:v', '1', '-vf', 'scale=1280:-2', '-pix_fmt', 'yuvj420p', '-q:v', '3', path)
                images.append(base64.b64encode(path.read_bytes()).decode())
            body = dict(model=args.model, prompt=PROMPT, images=images,
                        stream=False, format=SCHEMA, options=dict(temperature=0, num_ctx=8192))
            request = urllib.request.Request(args.ollama.rstrip('/') + '/api/generate',
                        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
            result = validate_result(json.loads(answer['response']))
            saved['windows'].append(dict(start=start, end=end, **result))
            write_json(cache, saved)
            print(f'{index + 1}/{count}: {start:.0f}s {result["event"]} {result["score"]:.2f}', flush=True)
    windows = saved['windows'][:count]
    clips = select_clips(windows, duration, args.top, args.minimum, args.padding)
    write_json(output / 'highlights.json', dict(video=str(video), duration=duration,
               analyzed_until=windows[-1]['end'] if windows else 0,
               complete=count == math.ceil(duration / args.window), clips=clips))
    lines = ['# Candidate highlights — unverified', '',
             'Model suggestions from sampled frames; review the footage before calling an event a goal.', '']
    lines += [f'- {clip["start"]:.1f}–{clip["end"]:.1f}s: {clip["event"]} '
              f'({clip["score"]:.2f}) — {clip["reason"]}' for clip in clips]
    if not clips:
        lines.append('No windows met the selection threshold.')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def summarize(args):
    """Analyze/checkpoint windows and render selected candidates in one command."""
    output = Path(args.output)
    reel = output / 'highlights.mp4'
    if reel.exists():
        raise ValueError('Highlight reel already exists; choose a new output directory or use analyze to resume analysis.')
    analyze(args)
    manifest = output / 'highlights.json'
    if not json.loads(manifest.read_text())['clips']:
        print(f'No candidates met the threshold; review {output / "summary.md"}.')
        return
    render(argparse.Namespace(manifest=str(manifest), video=args.video, output=str(reel)))


def render(args):
    manifest = json.loads(Path(args.manifest).read_text())
    video = Path(args.video or manifest['video']).resolve()
    duration = float(probe(video)['format']['duration'])
    clips = manifest['clips']
    if not clips:
        raise ValueError('No candidate highlights to render. Review the analysis first.')
    for clip in clips:
        if not 0 <= clip['start'] < clip['end'] <= duration + .1:
            raise ValueError('Invalid clip timestamps.')
    output = Path(args.output).resolve()
    if output == video or output.exists():
        raise ValueError('Choose a new output file; refusing to overwrite existing media.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        folder = Path(temporary)
        for index, clip in enumerate(clips):
            run('ffmpeg', '-v', 'error', '-ss', clip['start'], '-i', video,
                '-t', clip['end'] - clip['start'], '-map', '0:v:0', '-map', '0:a:0?',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-avoid_negative_ts', 'make_zero', folder / f'clip{index}.mp4')
        listing = folder / 'concat.txt'
        listing.write_text(''.join(f"file 'clip{index}.mp4'\n" for index in range(len(clips))))
        run('ffmpeg', '-v', 'error', '-n', '-f', 'concat', '-safe', '1', '-i', listing,
            '-c', 'copy', '-movflags', '+faststart', output)
    print(f'Saved {output}')


def dewarp(args):
    if args.start < 0 or args.duration <= 0 or not 0 < args.fov < 180:
        raise ValueError('Use nonnegative start, positive duration and FOV below 180 degrees.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    projection = (f'rotate={args.rotation}*PI/180,'
                  f'v360=input=fisheye:output={args.projection}:ih_fov={args.input_fov}:'
                  f'iv_fov={args.input_fov}:h_fov={args.fov}:v_fov={args.vertical_fov}:'
                  f'yaw={args.yaw}:pitch={args.pitch}:w=1280:h=720')
    run('ffmpeg', '-v', 'warning', '-n', '-ss', args.start, '-i', args.video,
        '-t', args.duration, '-map', '0:v:0', '-map', '0:a:0?', '-vf', projection,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-movflags', '+faststart', output)
    write_json(output.with_suffix('.json'), dict(source=str(Path(args.video).resolve()),
               start=args.start, duration=args.duration, filter=projection,
               note='Approximate lens model; original retained, no added source detail.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    from .doctor import add_command
    add_command(commands)
    command = commands.add_parser('download')
    command.add_argument('url')
    command.add_argument('--output', default=None)
    command.add_argument('--camera', default=None)
    command.set_defaults(function=download)
    command = commands.add_parser('analyze', aliases=['summarize'],
                                  help='Analyze match windows; summarize also renders a candidate reel')
    command.add_argument('video')
    command.add_argument('--output', default='outputs/analysis')
    command.add_argument('--ollama', default='http://127.0.0.1:11434')
    command.add_argument('--model', default='qwen3-vl:4b')
    command.add_argument('--window', type=float, default=12)
    command.add_argument('--frames', type=int, default=6)
    command.add_argument('--limit', type=int, help='Analyze only the first N windows for a pilot')
    command.add_argument('--top', type=int, default=10)
    command.add_argument('--minimum', type=float, default=.65)
    command.add_argument('--padding', type=float, default=4)
    command.set_defaults(function=analyze)
    command = commands.add_parser('render')
    command.add_argument('manifest')
    command.add_argument('--video', help='Override input path after moving to another machine')
    command.add_argument('--output', default='outputs/highlights.mp4')
    command.set_defaults(function=render)
    command = commands.add_parser('dewarp', help='Create an adjustable fisheye viewing preview')
    command.add_argument('video')
    command.add_argument('--output', default='outputs/dewarped.mp4')
    command.add_argument('--start', type=float, default=600)
    command.add_argument('--duration', type=float, default=20)
    command.add_argument('--projection', choices=['flat', 'cylindrical'], default='cylindrical')
    command.add_argument('--input-fov', type=float, default=180)
    command.add_argument('--fov', type=float, default=170)
    command.add_argument('--vertical-fov', type=float, default=100)
    command.add_argument('--rotation', type=float, default=-15)
    command.add_argument('--yaw', type=float, default=0)
    command.add_argument('--pitch', type=float, default=25)
    command.set_defaults(function=dewarp)
    from .postprocess import add_commands
    add_commands(commands)
    args = parser.parse_args()
    if args.command == 'summarize':
        args.function = summarize
    try:
        if args.command == 'doctor':
            parser.exit(args.function(args))
        prepare_storage()
        if args.command == 'download' and args.output is None:
            args.output = 'matches/' + hashlib.sha256(playlist_url(args.url, args.camera).encode()).hexdigest()[:16]
        args.output = str(output_path(args.output))
        for name in ('video', 'manifest', 'weights'):
            if getattr(args, name, None):
                setattr(args, name, str(input_path(getattr(args, name))))
        args.function(args)
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
