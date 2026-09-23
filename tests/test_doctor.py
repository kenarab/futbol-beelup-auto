import argparse
import contextlib
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

from futbol_beelup.doctor import doctor, local_url, cuda_check


class DoctorTests(unittest.TestCase):
    def test_diagnostics_without_storage_or_dependencies(self):
        args = argparse.Namespace(ollama='http://127.0.0.1:11434', model='missing',
                                  weights=None, smoke=False, json=True)
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), \
             patch('futbol_beelup.doctor.shutil.which', return_value=None), \
             patch('futbol_beelup.doctor.import_module', side_effect=ImportError('no torch')), \
             patch('futbol_beelup.doctor.urllib.request.urlopen', side_effect=OSError('offline')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(doctor(args), 1)
        report = json.loads(output.getvalue())
        checks = {item['name']: item for item in report['checks']}
        self.assertFalse(report['ready'])
        for name in ('storage', 'ffmpeg', 'torch_cuda', 'ollama_vision_model', 'weights'):
            self.assertEqual(checks[name]['status'], 'missing')

    def test_local_endpoint_only(self):
        for url in ('https://localhost:11434', 'http://example.com',
                    'http://user@localhost:11434', 'http://localhost/api'):
            with self.assertRaises(ValueError):
                local_url(url)
        self.assertEqual(local_url('http://127.0.0.1:11434/'), 'http://127.0.0.1:11434')

    def test_cpu_torch_does_not_claim_cuda_readiness(self):
        with patch('futbol_beelup.doctor.import_module') as load:
            load.return_value.__version__ = "test-cpu"
            load.return_value.cuda.is_available.return_value = False
            with self.assertRaisesRegex(ValueError, 'cannot access CUDA'):
                cuda_check(smoke=True)

    def test_cli_dispatch_does_not_require_storage(self):
        from futbol_beelup.cli import main
        with patch.object(sys, 'argv', ['futbol', 'doctor', '--json']), \
             patch('futbol_beelup.doctor.doctor', return_value=1) as run, \
             patch('futbol_beelup.cli.prepare_storage') as prepare:
            with self.assertRaises(SystemExit) as result:
                main()
            self.assertEqual(result.exception.code, 1)
            run.assert_called_once()
            prepare.assert_not_called()
