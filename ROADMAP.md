# GPU development roadmap

Goal: validate a local workflow that archives matches, proposes reviewable highlights,
and exports comfortable ball-following video.

## Baseline — 2026-09-22

- Implemented: download/archive, resumable highlight analysis, highlight rendering,
  dewarping, YOLO tracking, camera smoothing, and export resizing.
- Confirmed: RTX 3060 with 12,288 MiB VRAM and driver 575.57.08; Python 3.13.5;
  FFmpeg, ffprobe, and Ollama executables available.
- All 11 existing tests pass, including the real FFmpeg rendering test.
- Unverified: PyTorch CUDA inference, Ollama service/models, local detector weights,
  real-footage accuracy, throughput, and peak GPU memory. The CUDA version shown
  by nvidia-smi does not establish that a PyTorch runtime is installed.
- No .venv or archived data/ directory exists in this checkout. FUTBOL_DATA_DIR
  was not set in the inspected shell. Locate the previously rescued archive.

## 1. Reproducible GPU environment

- [x] Implement `futbol doctor`: report native tools, storage readiness, optional
  dependencies, CUDA device/VRAM, local weights, and Ollama connectivity. Work
  without GPU extras and explain missing prerequisites.
- [x] Create an isolated virtualenv with compatible Python/PyTorch versions,
  install GPU extras, and record the validated environment.
- [x] Configure a data root outside Git and download/verify the match archive
  and checksums. This host currently uses its internal disk.
- [ ] Supply local detector weights and prepare an installed Ollama vision model.
- [x] Verify actual CUDA computation, then run tests, package build, and pip check.

Exit: diagnostics show a ready environment, source footage decodes, and both
inference backends complete a small request. Record versions and commands.

## 2. Measured short-clip pilot

- [ ] Choose a 30–60 second clip with ball movement and an occlusion.
- [ ] Tune dewarping to retain both goals and save the transform settings.
- [ ] Run tracking, follow rendering, and five highlight-analysis windows sequentially.
- [ ] Record model identity, input dimensions/duration, settings, wall time,
  tracking throughput, peak VRAM, output sizes, and failures.
- [ ] Review ball positions, camera motion, highlight candidates, and audio sync.

Exit: reproducible commands, playable outputs, and a pilot report documenting
observed quality and limitations. Encoding success alone is insufficient.

## 3. Tracking evaluation and improvement

- [ ] Label clips covering near/far balls, frame edges, occlusions, and confusing
  pitch markings. Separate tuning footage from held-out evaluation footage.
- [ ] Add evaluation for precision/recall, normalized position error, missing
  detection intervals, and visible-ball crop retention.
- [ ] Record a baseline and agree numerical acceptance targets before tuning.
- [ ] Compare detector resolution, sampling rate, confidence, and motion gating.
  Consider specialized weights or fine-tuning if measured failures justify it.
- [ ] Improve temporal association/recovery while preserving explicit missing
  observations; do not label predictions as observed ball positions.

Exit: agreed targets are met on held-out clips with reproducible results.

## 4. Viewing quality and highlight usefulness

- [ ] Review jitter, excessive pans, goal visibility, and crop loss; introduce
  adaptive widening during uncertainty if the pilot demonstrates a need.
- [ ] Label known chances/goals and measure highlight event recall and false
  candidates; tune sampling and window settings against those annotations.
- [ ] Preserve source timestamps and mark generated candidates as unverified
  until reviewed.

Exit: reviewed clips meet agreed camera/highlight targets, preserve audio sync,
and remain traceable to the source match.

## 5. Full-match reliability and performance

- [ ] Process a complete match; record runtime, storage growth, and peak memory.
- [ ] Exercise interruption/resume, missing storage, insufficient space, and model
  failures. Extend checkpointing to expensive stages where needed.
- [ ] Save run manifests with source/model hashes, environment, parameters, outputs.
- [ ] Profile before adding batching, hardware encoding, or combined dewarp/crop
  rendering. Check quality after each optimization.

Exit: a complete match runs within measured hardware/storage limits and an
interrupted run recovers without corrupting existing outputs.

## Deferred

AI super-resolution, lens-aware 3D camera steering, training infrastructure, and
a web UI follow a validated baseline and a concrete demonstrated need.

## Next implementation

Development is paused at the user's request. On resumption, run the short-clip
vision pilot using the isolated Ollama runtime described below. Detector weights
and tracking evaluation remain pending. Keep large artifacts in the data root.

## Hardware session results — 2026-09-22

- Implemented `doctor` with optional CUDA computation and `summarize` for combined
  analysis/reel generation; 17 tests and package build passed.
- Created the project virtualenv. PyTorch 2.11.0+cu128 / CUDA runtime 12.8 completed
  an actual matrix multiplication on the RTX 3060.
- Python-to-Ollama inference with the existing qwen:0.5b model succeeded in 3.63 s;
  Ollama reported 1,940,098,048 bytes resident in VRAM. This tests the backend,
  not football understanding or vision accuracy.
- Downloaded all 15 source originals and combined match 35379808 to the configured
  data root: 4502.24 seconds, 1024x1024, with audio. Full combined decode passed;
  SHA-256 checksums for the 16 media files are saved in preservation.json.
- No external drive was mounted; this session explicitly configured an internal
  folder outside Git using the ignored .env file.
- Qwen3-VL download was rejected because installed Ollama 0.5.7 is too old.
  Upgrade the inference service before the vision pilot; no vision quality claims
  or full-match inference results are established by these checks.
- Local detector weights and a real tracking pilot remain outstanding.

## Pause checkpoint — 2026-09-23

- Installed isolated Ollama 0.34.3 under `$FUTBOL_DATA_DIR/tools/ollama`.
  The original system service (0.5.7) was not modified.
- Downloaded `qwen3-vl:4b` successfully into `$FUTBOL_DATA_DIR/models/ollama`:
  3,295,636,135 bytes, Q4_K_M, model digest
  `1343d82ebee38e26a4dd6b0180b915eb91550184e67c505dea97509571c8f683`.
- Stopped the isolated server on port 11435 at the user's request. No match
  inference was started. All 17 tests passed before pausing.
- Match media, runtime, models, and ignored local configuration remain on disk.

When development is resumed, start the isolated server in a separate terminal:

```sh
source .env
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS="$FUTBOL_DATA_DIR/models/ollama" \
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1 \
"$FUTBOL_DATA_DIR/tools/ollama/bin/ollama" serve
```

Then run the bounded pilot:

```sh
source .env
.venv/bin/futbol summarize matches/35379808/match.mp4 \
  --ollama http://127.0.0.1:11435 --model qwen3-vl:4b \
  --output outputs/35379808/recap-pilot --limit 5
```
