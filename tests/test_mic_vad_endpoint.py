"""One mic cue endpoint follows each qualifying utterance."""

import unittest

import numpy as np

from mic_vad import SileroMicFilter


class FakeVad:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def __call__(self, _frame, _sample_rate):
        return next(self.probabilities)


class MicEndpointTests(unittest.TestCase):
    def test_emits_once_after_speech_and_hangover(self):
        detector = SileroMicFilter(
            model=FakeVad([0.9] * 6 + [0.0] * 3),
            pre_roll_frames=0, hangover_frames=2,
        )
        for _ in range(6):
            detector.process(np.zeros(512, dtype=np.float32))
            self.assertEqual(detector.consume_speech_ends(), 0)
        detector.process(np.zeros(512, dtype=np.float32))
        self.assertEqual(detector.consume_speech_ends(), 0)
        detector.process(np.zeros(512, dtype=np.float32))
        self.assertEqual(detector.consume_speech_ends(), 1)
        detector.process(np.zeros(512, dtype=np.float32))
        self.assertEqual(detector.consume_speech_ends(), 0)

    def test_ignores_too_short_vad_burst(self):
        detector = SileroMicFilter(
            model=FakeVad([0.9] * 5 + [0.0] * 2),
            pre_roll_frames=0, hangover_frames=2,
        )
        for _ in range(7):
            detector.process(np.zeros(512, dtype=np.float32))
        self.assertEqual(detector.consume_speech_ends(), 0)


if __name__ == "__main__":
    unittest.main()
