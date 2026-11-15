"""The quiet radio loop stays out of queued speech and survives its seam."""

from pathlib import Path
import unittest

import numpy as np

from idle_audio import IdleAudioLoop


class IdleAudioTests(unittest.TestCase):
    def test_asset_loads_quietly_at_both_output_rates(self):
        path = Path(__file__).resolve().parents[1] / "assets" / "radio-static-idle.mp3"
        for rate in (24_000, 48_000):
            with self.subTest(rate=rate):
                loop = IdleAudioLoop.from_file(path, rate)
                rms = float(np.sqrt(np.mean(loop.samples * loop.samples)))
                self.assertGreater(rms, 0.001)
                self.assertLess(rms, 0.005)
                self.assertLessEqual(float(np.max(np.abs(loop.samples))), 0.081)
                self.assertLess(abs(float(loop.samples[0] - loop.samples[-1])), 0.01)
                self.assertGreater(loop.trimmed_leading_seconds, 0.1)
                self.assertGreater(loop.trimmed_trailing_seconds, 0.01)
                self.assertAlmostEqual(loop.crossfade_seconds, 0.06, places=2)
                window = rate // 50
                levels = np.asarray([
                    np.sqrt(np.mean(loop.samples[i:i + window] ** 2))
                    for i in range(0, len(loop.samples), window)
                ])
                self.assertGreater(float(levels.min() / np.median(levels)), 0.5)

    def test_static_fades_out_for_speech_and_back_in_afterward(self):
        rate = 24_000
        loop = IdleAudioLoop(np.ones(rate, dtype=np.float32), rate)
        idle = np.zeros(rate // 10, dtype=np.float32)
        loop.mix_into(idle, True)
        self.assertGreater(float(np.mean(idle[-200:])), 0.001)

        speech = np.full(rate // 10, 0.2, dtype=np.float32)
        loop.mix_into(speech, False)
        self.assertLess(abs(float(np.mean(speech[-200:])) - 0.2), 0.0001)

        idle_again = np.zeros(rate // 10, dtype=np.float32)
        loop.mix_into(idle_again, True)
        self.assertGreater(float(np.mean(idle_again[-200:])), 0.001)


if __name__ == "__main__":
    unittest.main()
