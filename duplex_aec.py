"""48 kHz duplex audio I/O with a playback reference for WebRTC AEC3."""

from collections import deque
import queue

import numpy as np
from pywebrtc_audio import AudioProcessor
from scipy.signal import resample_poly


STREAM_RATE = 48_000
STREAM_BLOCK = 4_800  # 100 ms on both input and output
ECHO_GUARD_FLOOR = 10 ** (-48 / 20)
REFERENCE_ACTIVE_FLOOR = 10 ** (-50 / 20)


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


class EchoGuard:
    """Reject weak AEC residue without closing the mic to a clear interruption."""

    def __init__(self):
        self.hold_blocks = 0
        self.suppressed_blocks = 0

    def filter(self, near, cleaned, reference):
        reference_active = rms(reference) > REFERENCE_ACTIVE_FLOOR
        self.hold_blocks = 8 if reference_active else max(0, self.hold_blocks - 1)
        if not self.hold_blocks:
            return cleaned
        clean_level = rms(cleaned)
        if clean_level < ECHO_GUARD_FLOOR:
            self.suppressed_blocks += 1
            return np.zeros_like(cleaned)
        # A strong independent voice survives AEC; suppress only when the
        # output was mostly removed and its small residue remains speech-like.
        if reference_active and clean_level < rms(near) * 0.18:
            self.suppressed_blocks += 1
            return np.zeros_like(cleaned)
        return cleaned


class TimedReference:
    """Match actual DAC frames to microphone ADC frames on one stream clock."""

    def __init__(self, sample_rate=STREAM_RATE):
        self.sample_rate = sample_rate
        self.frames = deque()

    def add(self, start_time, samples):
        self.frames.append((start_time, samples.copy()))
        cutoff = start_time - 3.0
        while self.frames and self.frames[0][0] + len(self.frames[0][1]) / self.sample_rate < cutoff:
            self.frames.popleft()

    def render(self, start_time, length):
        result = np.zeros(length, dtype=np.float32)
        matched = 0
        for frame_start, frame in self.frames:
            offset = round((frame_start - start_time) * self.sample_rate)
            dst_start = max(0, offset)
            dst_end = min(length, offset + len(frame))
            if dst_end <= dst_start:
                continue
            src_start = dst_start - offset
            result[dst_start:dst_end] = frame[src_start:src_start + dst_end - dst_start]
            matched += dst_end - dst_start
        return result, matched


class DuplexAecIO:
    """Keep the microphone live while removing the known speaker signal."""

    def __init__(self, mic_blocks, speaker_chunks, errors, transcriber, put_latest,
                 gate_event=None, gate_enabled=False, scope_frames=None, idle_loop=None,
                 mic_gain=1.0):
        self.mic_blocks = mic_blocks
        self.speaker_chunks = speaker_chunks
        self.errors = errors
        self.transcriber = transcriber
        self.put_latest = put_latest
        self.gate_event = gate_event
        self.gate_enabled = gate_enabled
        self.scope_frames = scope_frames
        self.idle_loop = idle_loop
        self.mic_gain = mic_gain
        self.aec = AudioProcessor(sample_rate=STREAM_RATE, echo_cancellation=True)
        self.reference = TimedReference()
        self.speech_reference = TimedReference()
        self.echo_guard = EchoGuard()
        self.completions = queue.Queue()
        self.current_samples = None
        self.current_offset = 0
        self.current_end_turn = False
        self.dropped_mic_blocks = 0
        self.reference_matched = 0
        self.reference_requested = 0
        self.reference_active_blocks = 0
        self.input_latency = 0.1
        self.missing_adc_timestamps = 0

    @property
    def playing(self):
        return self.current_samples is not None

    @staticmethod
    def speaker_samples(samples):
        return resample_poly(samples, 2, 1).astype(np.float32)

    def callback(self, indata, outdata, frames, time_info, status):
        try:
            if status:
                # PortAudio status is reported by the main loop, outside this callback.
                self.completions.put(("status", str(status)))
            near = indata[:, 0].copy()
            adc_time = time_info.inputBufferAdcTime
            if adc_time <= 0:
                # Windows MME reports 0 for inputBufferAdcTime. PortAudio's
                # currentTime and outputBufferDacTime still share a clock.
                adc_time = time_info.currentTime - self.input_latency
                self.missing_adc_timestamps += 1
            far, matched = self.reference.render(adc_time, frames)
            speech_far, _ = self.speech_reference.render(adc_time, frames)
            self.reference_matched += matched
            self.reference_requested += frames
            if rms(speech_far) > REFERENCE_ACTIVE_FLOOR:
                self.reference_active_blocks += 1
            cleaned = self.aec.process(near, far)
            if rms(speech_far) <= REFERENCE_ACTIVE_FLOOR and not self.echo_guard.hold_blocks:
                # AEC can distort near-end speech even with a zero far end.
                # Keep the mic bit-for-bit intact outside speaker playback.
                cleaned = near
            cleaned = self.echo_guard.filter(near, cleaned, speech_far)
            mic = resample_poly(cleaned, 1, 3).astype(np.float32)
            np.multiply(mic, self.mic_gain, out=mic)
            np.clip(mic, -1.0, 1.0, out=mic)
            if self.gate_enabled and self.gate_event is not None and self.gate_event.is_set():
                mic.fill(0)
            if self.put_latest(self.mic_blocks, mic):
                self.dropped_mic_blocks += 1
            if self.transcriber is not None:
                self.transcriber.submit(mic)

            outdata.fill(0)
            occupied = np.zeros(frames, dtype=np.bool_) if self.idle_loop is not None else None
            position = 0
            while position < frames:
                if self.current_samples is None:
                    try:
                        self.current_samples, self.current_end_turn = self.speaker_chunks.get_nowait()
                    except queue.Empty:
                        break
                    self.current_offset = 0
                available = len(self.current_samples) - self.current_offset
                count = min(frames - position, available)
                if count:
                    outdata[position:position + count, 0] = self.current_samples[
                        self.current_offset:self.current_offset + count
                    ]
                    if occupied is not None:
                        occupied[position:position + count] = True
                    self.current_offset += count
                    position += count
                if self.current_offset == len(self.current_samples):
                    if self.current_end_turn:
                        dac_end = time_info.outputBufferDacTime + position / STREAM_RATE
                        self.completions.put(("turn", dac_end))
                    self.current_samples = None
            speech_output = outdata[:, 0].copy()
            if self.scope_frames is not None:
                try:
                    self.scope_frames.put_nowait(speech_output)
                except queue.Full:
                    pass
            if self.idle_loop is not None:
                self.idle_loop.mix_into(outdata[:, 0], ~occupied)
            self.speech_reference.add(time_info.outputBufferDacTime, speech_output)
            self.reference.add(time_info.outputBufferDacTime, outdata[:, 0])
        except Exception as exc:
            if self.errors.empty():
                self.errors.put_nowait(exc)
            outdata.fill(0)
