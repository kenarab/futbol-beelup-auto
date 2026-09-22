import unittest
from futbol_beelup.cli import original_urls, playlist_url, select_clips, validate_result


class PrototypeTests(unittest.TestCase):
    def test_full_match_and_camera(self):
        url = playlist_url('https://beelup.com/player.php?id=123&c=central')
        self.assertIn('tipo=todo', url)
        self.assertIn('camara=central', url)
        self.assertIn('camara=der', playlist_url('https://beelup.com/player.php?id=123', 'der'))
        with self.assertRaises(ValueError):
            playlist_url('https://beelup.com/player.php?id=bad')

    def test_archive_deduplicates_preserving_order(self):
        manifest = '#EXTM3U\nhttps://example.com/hls/a.mp4/seg1.ts\nhttps://example.com/hls/a.mp4/seg2.ts\nhttps://example.com/hls/b.mp4/seg1.ts'
        self.assertEqual(original_urls(manifest), ['https://example.com/a.mp4', 'https://example.com/b.mp4'])

    def test_selection_bounds_overlap_and_uncertainty(self):
        windows = [dict(start=0, end=12, score=.8, event='chance'),
                   dict(start=12, end=24, score=.9, event='save'),
                   dict(start=24, end=36, score=.99, event='unclear'),
                   dict(start=36, end=40, score=.7, event='chance')]
        clips = select_clips(windows, 40, 10, .65, 4)
        self.assertEqual([(c['start'], c['end']) for c in clips], [(8, 28), (32, 40)])
        self.assertEqual(select_clips(windows, 40, 10, 1, 4), [])

    def test_reject_invalid_model_scores(self):
        for score in (float('nan'), True, -1, 2, '0.9'):
            with self.assertRaises(ValueError):
                validate_result(dict(score=score, event='chance', reason='test'))


if __name__ == '__main__':
    unittest.main()
