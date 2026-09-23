import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from futbol_beelup.cli import summarize


class SummaryTests(unittest.TestCase):
    def test_render_only_when_candidates_exist(self):
        for clips in ([], [dict(start=0, end=5)]):
            with self.subTest(clips=clips), tempfile.TemporaryDirectory() as folder:
                args = argparse.Namespace(output=folder, video='/source.mp4')
                def analysis(_):
                    (Path(folder) / 'highlights.json').write_text(json.dumps(dict(clips=clips)))
                with patch('futbol_beelup.cli.analyze', side_effect=analysis) as analyze, \
                     patch('futbol_beelup.cli.render') as render:
                    summarize(args)
                    analyze.assert_called_once_with(args)
                    self.assertEqual(render.call_count, bool(clips))
                    if clips:
                        self.assertEqual(render.call_args.args[0].output, str(Path(folder) / 'highlights.mp4'))

    def test_existing_reel_is_protected_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'highlights.mp4').write_bytes(b'existing')
            with patch('futbol_beelup.cli.analyze') as analyze:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    summarize(argparse.Namespace(output=folder))
                analyze.assert_not_called()
