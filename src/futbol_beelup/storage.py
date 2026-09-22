"""Explicit external data root; never silently fall back to the checkout."""
import os
import shutil
from pathlib import Path


def data_root():
    value = os.environ.get('FUTBOL_DATA_DIR')
    if not value:
        raise ValueError('Set FUTBOL_DATA_DIR to an existing data folder outside the repository.')
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError('FUTBOL_DATA_DIR must be an absolute path.')
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Data folder unavailable: {root}. Mount the drive first; no fallback will be created.')
    if any((parent / '.git').exists() for parent in (root, *root.parents)):
        raise ValueError('FUTBOL_DATA_DIR must be outside the repository.')
    return root


def output_path(value):
    root = data_root()
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Output must be inside FUTBOL_DATA_DIR.')
    return path


def input_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else data_root() / path).resolve()


def prepare_storage():
    root = data_root()
    free = shutil.disk_usage(root).free
    if free < 512 * 1024**2:
        raise ValueError('Less than 512 MiB free on the data drive. Free space before processing.')
    # Keep third-party scratch files/caches off the system drive as well.
    for name, subdir in [('TMPDIR', 'tmp'), ('TEMP', 'tmp'), ('TMP', 'tmp'),
                         ('YOLO_CONFIG_DIR', 'cache/ultralytics'),
                         ('TORCH_HOME', 'cache/torch'), ('HF_HOME', 'cache/huggingface')]:
        folder = root / subdir
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    import tempfile
    tempfile.tempdir = str(root / 'tmp')
    print(f'Data root: {root} ({free / 2**30:.1f} GiB free)', flush=True)
    return root
