import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from futbol_beelup.cli import probe, write_json
from futbol_beelup.postprocess import camera_path, choose_ball, fingerprint, follow
from futbol_beelup.storage import data_root, output_path


class StorageTests(unittest.TestCase):
    def test_requires_explicit_available_root(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            data_root()
        with patch.dict(os.environ, {'FUTBOL_DATA_DIR': '/not-a-mounted-drive/futbol'}), self.assertRaises(ValueError):
            data_root()

    def test_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'FUTBOL_DATA_DIR': directory}):
            self.assertEqual(output_path('outputs/reel.mp4'), Path(directory).resolve()/'outputs/reel.mp4')
            with self.assertRaises(ValueError):
                output_path('../oops.mp4')
            link = Path(directory)/'escape'
            link.symlink_to(Path(directory).parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                output_path('escape/oops.mp4')

    def test_rejects_repository_root(self):
        repo = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {'FUTBOL_DATA_DIR': str(repo)}), self.assertRaises(ValueError):
            data_root()


class TrackingTests(unittest.TestCase):
    def test_reject_jump(self):
        self.assertEqual(choose_ball([(.9,.9,.99),(.51,.5,.6)],(.5,.5,.8),.1),(.51,.5,.6))

    def test_smoothing_and_missing_ball(self):
        path = list(camera_path([dict(t=0,ball=[.95,.5,.9]),dict(t=.5,ball=None)],4))
        self.assertTrue(all(abs(b[1]-a[1]) <= .35/25+1e-9 for a,b in zip(path,path[1:])))
        self.assertGreater(path[15][1],.5)
        self.assertLess(abs(path[-1][1]-.5),.03)
        self.assertTrue(all(0<=row[1]<=1 and 0<=row[2]<=1 for row in path))

    def test_invalid_track(self):
        with self.assertRaises(ValueError):
            list(camera_path([dict(t=0,ball=[math.nan,.5,.9])],1))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_camera_commands_audio_and_upscale(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video, manifest, output = folder/'source.mp4',folder/'track.json',folder/'follow.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=10',
                            '-f','lavfi','-i','sine=frequency=440','-t','2','-c:v','libx264','-c:a','aac',str(video)],check=True)
            write_json(manifest,dict(video=str(video),sha256=fingerprint(video),start=0,duration=2,width=320,height=180,
                       samples=[dict(t=0,ball=[.8,.5,.9]),dict(t=1,ball=[.3,.5,.9])]))
            follow(argparse.Namespace(manifest=str(manifest),video=None,output=str(output),crop=.8,fps=10,resolution='720p'))
            info=probe(output)
            stream=next(s for s in info['streams'] if s['codec_type']=='video')
            self.assertEqual((stream['width'],stream['height']),(1280,720))
            self.assertTrue(any(s['codec_type']=='audio' for s in info['streams']))
            self.assertAlmostEqual(float(info['format']['duration']),2,delta=.2)
            self.assertFalse(any(p.is_dir() for p in folder.iterdir()))
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'],check=True)
            with self.assertRaises(ValueError):
                follow(argparse.Namespace(manifest=str(manifest),video=None,output=str(output),crop=.8,fps=10,resolution='720p'))
