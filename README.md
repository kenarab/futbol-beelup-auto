# Futbol Beelup POC

Python package for archiving football matches, proposing highlights with a local
vision model, dewarping fisheye video, and testing ball-following exports.
Development is on this Mac. The initial GPU target is **Ubuntu, RTX 3060, 12 GB VRAM**;
storage, detector and compute device remain configurable.

## Standard project layout

```text
pyproject.toml                 metadata, dependencies and futbol entry point
src/futbol_beelup/
    __init__.py
    __main__.py               python -m futbol_beelup
    cli.py                    download, analyze, render, dewarp
    storage.py                data root, scratch paths and caches
    postprocess.py            ball detection, camera smoothing, export scaling
tests/                        unit tests and FFmpeg integration test
.env.example                  shell configuration example
.venv/                        isolated development environment (Git-ignored)
```

Legacy `data/` and `outputs/` are Git-ignored. The rescue download remains there
until the external drive is identified; new processing writes outside the repo.

## Install using a virtualenv

Use Python 3.10+. On Ubuntu, Python 3.11/3.12 is a starting point for the GPU stack,
subject to the selected PyTorch build. Never install this package with system pip.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -e '.[dev]'
futbol --help
```

`yt-dlp` is installed in this virtualenv and invoked through the same interpreter.
FFmpeg, ffprobe and curl are native executables required on PATH, not Python
packages. Ollama is a separate local service needed only for highlight analysis.
The base package does not install GPU dependencies or download model weights.
The `dev` extra adds Python package build tools.

On the Ubuntu GPU box, create its own virtualenv and install the CUDA-compatible
PyTorch build using the [official installer selector](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install --no-cache-dir -e '.[gpu]'
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

The optional extra provides Ultralytics/OpenCV. Supply detector weights as a local
file; the tracking command does not fetch them automatically. Device `0` selects
the first NVIDIA GPU; `cpu` is available too. See the
[detector API](https://docs.ultralytics.com/modes/predict).
Dependencies are bounded rather than locked across platforms; freeze a validated
GPU environment after the first hardware pilot. No GPU dependencies were installed
into this Mac's system Python.

## Check this machine

```sh
futbol doctor --json
futbol doctor --smoke --weights models/ball-detector.pt
```

Diagnostics run even when storage or optional GPU packages are missing. Exit code
1 means at least one full-pipeline prerequisite is missing; individual commands
may still work. `--smoke` performs an actual PyTorch CUDA matrix multiplication.
Checking weights confirms file existence, not detector accuracy or compatibility.

## External data root

Choose an existing folder on the mounted drive, then:

```sh
# Mac example — replace this with the actual mounted drive/folder:
export FUTBOL_DATA_DIR="/Volumes/MyVideoDrive/futbol"
# Ubuntu example:
# export FUTBOL_DATA_DIR="/media/$USER/MyVideoDrive/futbol"
```

The variable holds the directory path, not video contents. Alternatively copy
`.env.example` to `.env`, edit it and run `source .env`. Python does not load `.env`
automatically. Set the variable in each session or your own shell configuration.
No machine-specific path is committed.

- Root must exist, be absolute and be outside a Git repository.
- Missing/unmounted root stops processing: no silent local fallback.
- Relative input/output paths resolve inside the root, not the checkout.
- Absolute inputs may be elsewhere, including the existing rescue download.
- All outputs must stay inside the root; escaping through symlinks is rejected.
- Analysis/export temporary files live beside outputs; other scratch and model
  library caches are redirected to `tmp/` and `cache/` within the root.
- Startup rejects less than 512 MiB free. This is a minimum, not a guarantee that
  the entire job will fit. Allow room for source and final video files.

Suggested storage:

```text
$FUTBOL_DATA_DIR/
    matches/35379808/match.mp4
    matches/35379808/originals/
    models/ball-detector.pt
    outputs/35379808/
    tmp/
    cache/
```

The code accepts an external disk or pendrive. Test a short export to assess the
actual device. It streams frames instead of creating thousands of image files;
final compressed video still requires disk writes. Do not unplug during a job.
Ollama model storage is separate: configure `OLLAMA_MODELS` in the **Ollama service's
environment** if desired. This CLI cannot relocate a running service's models.

## Preserve the match

```sh
futbol download 'https://beelup.com/player.php?id=35379808' --output matches/35379808
```

Use a different directory per match/camera; `--camera` is optional. If no output
is provided, a stable URL hash is used under `matches/`. Beelup downloads archive
every source MP4 unchanged, save metadata, and join `match.mp4` without video
re-encoding. Only the combined copy removes H.264 filler NAL units that caused
MP4 decoding errors on this footage. Other URLs use yt-dlp's default best-format
selection and produce `full-match.mp4`.

An incomplete original file restarts on retry; completed originals are reused.
Missing files fail the command. Combined duration is checked against the playlist.
Partial-match URLs whose original-file boundaries differ may fail this check;
this POC prioritizes full matches. Downloads require a still-accessible source.

The rescued match remains at `data/35379808/match.mp4` in this checkout: **75:02,
1024×1024, with audio**, alongside all 15 originals and SHA-256 checksums in
`preservation.json`. This is the highest quality exposed by the inspected page and
playlist. Full decode verification passed. Copy and verify the archive on the
external drive before removing the local copy. No migration has yet been performed.

Optional full verification:

```sh
ffmpeg -v error -xerror -i "$FUTBOL_DATA_DIR/matches/35379808/match.mp4" -map 0:v:0 -map '0:a:0?' -fps_mode passthrough -enc_time_base demux -f null -
```

## Local highlight pilot

On Ubuntu, install/start Ollama, then:

```sh
ollama pull qwen3-vl:4b
futbol analyze matches/35379808/match.mp4 --output outputs/35379808/analysis --limit 5
```

The RTX 3060 is listed in [Ollama GPU support](https://docs.ollama.com/gpu). The
[4B vision model](https://ollama.com/library/qwen3-vl) is a starting point for VRAM
headroom, not a benchmark guarantee. Check `ollama ps` during the pilot. `--model`
selects another installed vision model. After model download, inference is local.

Remove `--limit 5` to resume the whole match. Each completed window is checkpointed;
a different video/model/sampling setup needs a new output directory. Six frames
per 12 seconds are sampled by default, which can miss fast shots and tiny balls.
Review `windows.json`, `highlights.json` and `summary.md` as **unverified candidates**.

```sh
futbol render outputs/35379808/analysis/highlights.json --output outputs/35379808/highlights.mp4
```

`--video` overrides a saved source path after moving between machines. Actual
highlight quality and GPU performance still need validation on the Ubuntu box.

## Automatic match recap

```sh
futbol summarize matches/35379808/match.mp4 --output outputs/35379808/recap --limit 5
```

This runs the existing frame analysis and creates `windows.json`, `highlights.json`,
`summary.md`, and (when candidates qualify) `highlights.mp4` in one command. Omit
`--limit` for the full match. The written summary describes candidate events, not a
verified scoreline. No qualifying events means no reel is generated.

Interrupted analysis resumes completed windows with the same video/model/settings.
An existing reel is protected from overwrite: use `analyze` to extend a pilot's
analysis, then `render` with a fresh output filename, or choose a new directory.
Rendering itself is not checkpointed.

The starting model is [Qwen3-VL 4B](https://ollama.com/library/qwen3-vl), a local
vision-language model. It requires a newer Ollama than this host's initial 0.5.7.
For this implementation, sampled frames provide evidence for candidate events;
small or obscured balls and fast actions can be missed. A later version can combine
ball tracking and denser temporal analysis with these proposals. Soccer-specific
[action spotting](https://www.soccer-net.org/tasks/action-spotting) is another
research direction, but broadcast-footage results need validation on this fisheye
amateur footage before adopting a model.

CUDA accelerates NVIDIA model inference. Ollama manages its own inference runtime;
YOLO tracking uses CUDA-enabled PyTorch. Downloading and the current FFmpeg exports
do not require CUDA. We use prebuilt runtimes and do not need custom CUDA kernels.

## Fisheye correction, then ball-following

Start with a short dewarped clip retaining both goals:

```sh
futbol dewarp matches/35379808/match.mp4 --start 600 --duration 30 --output outputs/35379808/wide.mp4
futbol track outputs/35379808/wide.mp4 --weights models/ball-detector.pt --device 0 --duration 30 --output outputs/35379808/track.json
futbol follow outputs/35379808/track.json --crop 0.8 --resolution 1080p --output outputs/35379808/follow-1080p.mp4
```

Dewarp defaults are approximate cylindrical lens settings. Adjust `--rotation`,
`--yaw`, `--pitch`, `--input-fov`, `--fov` and `--vertical-fov`; `--projection flat`
is a narrower perspective. Wide views stretch edges; narrow views can crop goals.
The output is a separate viewing copy plus transform metadata.

Tracking uses a local YOLO-compatible detector. Default class is `sports ball`;
use `--ball-class ball` for a custom model with that label. Frames are streamed and
sampled at 8 FPS, without writing frame images. Confidence and a motion gate reject
some false detections. `track.json` records normalized positions and explicit nulls
when the ball is not found. Generic models may miss this tiny ball or detect pitch
markings instead. Actual match tracking has not been validated on the GPU.

The virtual camera applies a dead zone and speed-limited smoothing. It holds briefly
when the ball disappears, then returns toward center. It does not invent observed
ball positions. `--crop` controls retained width; begin wide. It does not yet zoom
out during occlusions or perform lens-aware 3D camera steering. The source hash
prevents using a track with a different video.

## Upscaling strategy and disk use

Detect before upscaling → smooth camera motion → crop → resize once at final export.
`--resolution native` is the default; `720p` and `1080p` use
[FFmpeg Lanczos interpolation](https://ffmpeg.org/ffmpeg-scaler.html).
This changes display sizing, not true source detail. AI super-resolution is deferred
until we can measure temporal flicker and invented details. No AI upscaler is bundled.

Follow rendering writes a small text camera path and a temporary compressed MP4
on the chosen drive, preserves audio and crops/scales in one encode. The preceding
dewarped clip is one intermediate video. Fusing dewarp and camera rendering is a
later quality/storage optimization. Initially run detection and export sequentially
to avoid competing for the RTX 3060's memory.

## Development checks

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m build
python -m pip check
```

Tests use tiny synthetic media and cover source parsing, highlight selection,
missing drives, path escapes, camera smoothing/occlusion, and real FFmpeg rendering
with audio and 720p scaling. They do not establish detector accuracy. Label several
known ball positions/chances and inspect the short Ubuntu GPU pilot before running
the full match. The package follows the standard
[PyPA src-layout workflow](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
Beelup extraction depends on its current player implementation and may change.
