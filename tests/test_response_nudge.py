import unittest

import numpy as np

from response_nudge import ResponseNudge


class ResponseNudgeTests(unittest.TestCase):
    def test_only_biases_after_speech_ends_and_stops_after_two_attempts(self):
        nudge = ResponseNudge()
        logits = np.array([[0.0, 5.0]], dtype=np.float32)
        self.assertEqual(nudge.before_generate(-160, logits, 1), 0)
        self.assertEqual(nudge.before_generate(-24, logits, 1), 0)
        self.assertEqual(logits[0, 1], 5.0)
        self.assertEqual(nudge.before_generate(-160, logits, 1), 1)
        self.assertEqual(logits[0, 1], 2.0)
        self.assertFalse(nudge.after_generate(True))
        self.assertEqual(nudge.before_generate(-160, logits, 1), 2)
        self.assertTrue(nudge.after_generate(True))
        self.assertEqual(nudge.before_generate(-160, logits, 1), 0)

    def test_reply_clears_pending_nudge(self):
        nudge = ResponseNudge()
        logits = np.array([[0.0, 5.0]], dtype=np.float32)
        nudge.before_generate(-30, logits, 1)
        self.assertFalse(nudge.after_generate(False))
        self.assertEqual(nudge.before_generate(-160, logits, 1), 0)


if __name__ == "__main__":
    unittest.main()
