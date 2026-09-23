"""Readiness checks that also run before storage or GPU extras are configured."""
from importlib import import_module
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

from .storage import data_root, input_path


def local_url(value):
    parsed = urllib.parse.urlparse(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('Use a local Ollama HTTP URL, for example http://127.0.0.1:11434.')
    return value.rstrip('/')


def cuda_check(smoke=False):
    torch = import_module('torch')
    if not torch.cuda.is_available():
        raise ValueError(f'PyTorch {torch.__version__} cannot access CUDA (build: {torch.version.cuda}).')
    detail = dict(version=torch.__version__, runtime=torch.version.cuda,
                  device=torch.cuda.get_device_name(0), smoke_test='not requested')
    if smoke:
        matrix = torch.ones((32, 32), device='cuda:0')
        result = matrix @ matrix
        torch.cuda.synchronize()
        if not torch.all(result == 32).item():
            raise ValueError('CUDA matrix multiplication returned an incorrect result.')
        detail['smoke_test'] = 'passed'
    return detail


def collect(args):
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append(dict(name=name, status='ok', detail=detail))
        except Exception as error:
            checks.append(dict(name=name, status='missing', detail=str(error)))

    def executable(name):
        path = shutil.which(name)
        if not path:
            raise ValueError(f'Install {name} and make it available on PATH.')
        return path

    def storage():
        root = data_root()
        free = shutil.disk_usage(root).free
        if free < 512 * 1024**2:
            raise ValueError('Less than 512 MiB free on the data drive.')
        return dict(path=str(root), free_gib=round(free / 2**30, 2))

    def nvidia():
        result = subprocess.run([executable('nvidia-smi'),
            '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()

    def ollama():
        url = local_url(args.ollama)
        with urllib.request.urlopen(url + '/api/tags', timeout=5) as response:
            models = [item['name'] for item in json.load(response)['models']]
        if args.model not in models:
            raise ValueError(f'{args.model} is not installed. Installed models: {", ".join(models) or "none"}.')
        return dict(model=args.model, endpoint=url)

    check('storage', storage)
    for name in ('ffmpeg', 'ffprobe', 'curl'):
        check(name, lambda name=name: executable(name))
    check('nvidia', nvidia)
    check('torch_cuda', lambda: cuda_check(args.smoke))
    # Query package metadata without importing libraries that may initialize caches.
    from importlib.metadata import version
    for name in ('yt-dlp', 'opencv-python', 'ultralytics'):
        check(name, lambda name=name: version(name))
    check('ollama_vision_model', ollama)
    if args.weights:
        def weights():
            path = input_path(args.weights)
            if not path.is_file():
                raise ValueError(f'Detector weights not found: {path}')
            return dict(path=str(path), note='File exists; model compatibility is not tested.')
        check('weights', weights)
    else:
        checks.append(dict(name='weights', status='missing', detail='Pass --weights with a local detector file.'))
    return dict(python=sys.version.split()[0], executable=sys.executable,
                ready=all(item['status'] == 'ok' for item in checks), checks=checks)


def doctor(args):
    report = collect(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f'Python {report["python"]}: {report["executable"]}')
        for item in report['checks']:
            print(f'{item["status"].upper():7} {item["name"]}: {item["detail"]}')
        print('Full pipeline ready.' if report['ready'] else 'Some pipeline prerequisites are missing; see above.')
    return 0 if report['ready'] else 1


def add_command(commands):
    command = commands.add_parser('doctor', help='Check storage, models, and GPU prerequisites')
    command.add_argument('--ollama', default='http://127.0.0.1:11434')
    command.add_argument('--model', default='qwen3-vl:4b')
    command.add_argument('--weights')
    command.add_argument('--smoke', action='store_true', help='Run a small actual PyTorch CUDA computation')
    command.add_argument('--json', action='store_true', help='Print a machine-readable readiness report')
    command.set_defaults(function=doctor)
