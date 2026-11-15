"""Regression checks for the Windows duplex echo reference."""

import queue
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.signal import resample_poly

from duplex_aec import DuplexAecIO, EchoGuard, STREAM_BLOCK
from live_mic_duplex import scope_frame


class DuplexAecTests(unittest.TestCase):
    def test_microphone_gain_reduces_level_by_two_db(self):
        microphone = queue.Queue()
        errors = queue.Queue()
        gain = 10 ** (-2 / 20)
        audio = DuplexAecIO(
            microphone, queue.Queue(), errors, None,
            lambda target, block: target.put_nowait(block) or False,
            mic_gain=gain,
        )
        t = np.arange(STREAM_BLOCK, dtype=np.float32) / 48_000
        near = (0.08 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio.callback(
            near.reshape(-1, 1), np.zeros((STREAM_BLOCK, 1), dtype=np.float32),
            STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        expected = resample_poly(near, 1, 3).astype(np.float32) * gain
        np.testing.assert_allclose(microphone.get_nowait(), expected, atol=1e-7)
        self.assertTrue(errors.empty())

    def test_microphone_is_untouched_without_speaker_audio(self):
        microphone = queue.Queue()
        errors = queue.Queue()
        audio = DuplexAecIO(
            microphone, queue.Queue(), errors, None,
            lambda target, block: target.put_nowait(block) or False,
        )
        t = np.arange(STREAM_BLOCK, dtype=np.float32) / 48_000
        near = (0.08 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio.callback(
            near.reshape(-1, 1), np.zeros((STREAM_BLOCK, 1), dtype=np.float32),
            STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        np.testing.assert_allclose(
            microphone.get(), resample_poly(near, 1, 3).astype(np.float32),
        )
        self.assertTrue(errors.empty())

    def test_zero_adc_timestamp_uses_stream_clock(self):
        microphone = queue.Queue()
        speaker = queue.Queue()
        errors = queue.Queue()
        audio = DuplexAecIO(
            microphone, speaker, errors, None,
            lambda target, block: target.put_nowait(block) or False,
        )
        speaker.put((np.full(STREAM_BLOCK, 0.05, dtype=np.float32), True))
        input_block = np.zeros((STREAM_BLOCK, 1), dtype=np.float32)
        output_block = np.zeros_like(input_block)
        audio.callback(
            input_block, output_block, STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        audio.callback(
            input_block, output_block, STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.3,
                            outputBufferDacTime=1.5), None,
        )
        self.assertEqual(audio.missing_adc_timestamps, 2)
        self.assertEqual(audio.reference_matched, STREAM_BLOCK)
        self.assertTrue(errors.empty())
        self.assertEqual(microphone.get().shape, (STREAM_BLOCK // 3,))

    def test_echo_residue_is_suppressed_but_clear_voice_passes(self):
        guard = EchoGuard()
        reference = np.full(STREAM_BLOCK, 0.05, dtype=np.float32)
        residue = np.full(STREAM_BLOCK, 0.003, dtype=np.float32)
        voice = np.full(STREAM_BLOCK, 0.04, dtype=np.float32)
        self.assertFalse(guard.filter(reference, residue, reference).any())
        self.assertTrue(guard.filter(voice, voice, reference).any())
        self.assertFalse(guard.filter(residue, residue, np.zeros_like(reference)).any())

    def test_scope_telemetry_tracks_speaker_output(self):
        microphone = queue.Queue()
        speaker = queue.Queue()
        errors = queue.Queue()
        frames = queue.Queue(maxsize=4)
        audio = DuplexAecIO(
            microphone, speaker, errors, None,
            lambda target, block: target.put_nowait(block) or False,
            scope_frames=frames,
        )
        speaker.put((np.full(STREAM_BLOCK, 0.08, dtype=np.float32), True))
        output = np.zeros((STREAM_BLOCK, 1), dtype=np.float32)
        audio.callback(
            np.zeros_like(output), output, STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        np.testing.assert_allclose(frames.get_nowait(), output[:, 0])
        message = scope_frame(output[:, 0], np)
        self.assertTrue(message.startswith("[scope] 0.0800 "))
        self.assertEqual(len(message.split(" ", 2)[2].split(",")), 48)
        self.assertIsNone(scope_frame(np.zeros(STREAM_BLOCK, dtype=np.float32), np))
        self.assertTrue(errors.empty())

    def test_idle_bed_only_fills_unoccupied_output_and_enters_aec_reference(self):
        class FixedBed:
            def mix_into(self, output, idle_mask):
                output[idle_mask] += 0.02

        microphone = queue.Queue()
        speaker = queue.Queue()
        errors = queue.Queue()
        frames = queue.Queue()
        audio = DuplexAecIO(
            microphone, speaker, errors, None,
            lambda target, block: target.put_nowait(block) or False,
            scope_frames=frames, idle_loop=FixedBed(),
        )
        speaker.put((np.full(STREAM_BLOCK // 2, 0.1, dtype=np.float32), False))
        output = np.zeros((STREAM_BLOCK, 1), dtype=np.float32)
        audio.callback(
            np.zeros_like(output), output, STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        np.testing.assert_allclose(output[:STREAM_BLOCK // 2, 0], 0.1)
        np.testing.assert_allclose(output[STREAM_BLOCK // 2:, 0], 0.02)
        scope = frames.get_nowait()
        np.testing.assert_allclose(scope[:STREAM_BLOCK // 2], 0.1)
        np.testing.assert_allclose(scope[STREAM_BLOCK // 2:], 0.0)
        reference, matched = audio.reference.render(1.2, STREAM_BLOCK)
        self.assertEqual(matched, STREAM_BLOCK)
        np.testing.assert_allclose(reference, output[:, 0])
        speech_reference, _ = audio.speech_reference.render(1.2, STREAM_BLOCK)
        np.testing.assert_allclose(speech_reference, scope)
        self.assertTrue(errors.empty())

    def test_idle_static_does_not_keep_mic_echo_guard_active(self):
        class FixedBed:
            def mix_into(self, output, _idle_mask):
                output += 0.02

        microphone = queue.Queue()
        errors = queue.Queue()
        audio = DuplexAecIO(
            microphone, queue.Queue(), errors, None,
            lambda target, block: target.put_nowait(block) or False,
            idle_loop=FixedBed(),
        )
        near = np.full((STREAM_BLOCK, 1), 0.04, dtype=np.float32)
        audio.callback(
            np.zeros_like(near), np.zeros_like(near), STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=0.0, currentTime=1.0,
                            outputBufferDacTime=1.2), None,
        )
        microphone.get_nowait()
        audio.callback(
            near, np.zeros_like(near), STREAM_BLOCK,
            SimpleNamespace(inputBufferAdcTime=1.2, currentTime=1.3,
                            outputBufferDacTime=1.5), None,
        )
        np.testing.assert_allclose(
            microphone.get_nowait(), resample_poly(near[:, 0], 1, 3).astype(np.float32),
        )
        self.assertEqual(audio.echo_guard.hold_blocks, 0)
        self.assertTrue(errors.empty())


if __name__ == "__main__":
    unittest.main()
