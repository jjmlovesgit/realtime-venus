"""The key-off sound stays separate from generated reply audio."""

from pathlib import Path
import unittest

import numpy as np

from live_mic_duplex import load_playback_cue, queue_reply_ending


class ReplyCueTests(unittest.TestCase):
    def test_mp3_decodes_to_playback_rate(self):
        path = Path(__file__).resolve().parents[1] / "assets" / "mic-off.mp3"
        cue = load_playback_cue(path, np)
        self.assertEqual(cue.dtype, np.float32)
        self.assertEqual(cue.ndim, 1)
        self.assertGreater(len(cue), 12_000)
        self.assertLess(len(cue), 24_000)
        self.assertGreater(float(np.max(np.abs(cue))), 0.01)

    def test_cue_plays_before_tail_but_is_excluded_from_reply_wav(self):
        calls = []
        cue = np.ones(100, dtype=np.float32)
        tail = np.zeros(50, dtype=np.float32)

        def enqueue(samples, **kwargs):
            calls.append((samples, kwargs))

        queue_reply_ending(enqueue, cue, tail)
        self.assertIs(calls[0][0], cue)
        self.assertEqual(calls[0][1], {"include_in_debug": False})
        self.assertIs(calls[1][0], tail)
        self.assertEqual(calls[1][1], {"end_turn": True})


if __name__ == "__main__":
    unittest.main()
