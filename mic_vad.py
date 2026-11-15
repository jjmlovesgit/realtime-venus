"""Streaming Silero voice activity filter for 16 kHz microphone audio."""

from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad


class SileroMicFilter:
    SAMPLE_RATE = 16_000
    FRAME_SAMPLES = 512

    def __init__(self, threshold=0.6, pre_roll_frames=5, hangover_frames=10, model=None):
        self.threshold = threshold
        self.pre_roll_frames = pre_roll_frames
        self.hangover_frames = hangover_frames
        self.model = model if model is not None else load_silero_vad(onnx=True)
        self.pending = np.empty(0, dtype=np.float32)
        self.delayed = deque()
        self.hold_frames = 0
        self.speech_frames = 0
        self.in_utterance = False
        self.speech_end_events = 0

    def consume_speech_ends(self):
        """Return completed utterances since the last audio block."""
        count = self.speech_end_events
        self.speech_end_events = 0
        return count

    def process(self, samples):
        """Return speech plus lead-in/tail, replacing other samples with silence."""
        self.pending = np.concatenate((self.pending, np.asarray(samples, dtype=np.float32)))
        output = []
        while len(self.pending) >= self.FRAME_SAMPLES:
            frame = self.pending[: self.FRAME_SAMPLES].copy()
            self.pending = self.pending[self.FRAME_SAMPLES :]
            with torch.inference_mode():
                probability = float(self.model(torch.from_numpy(frame), self.SAMPLE_RATE))
            if probability >= self.threshold:
                self.hold_frames = self.hangover_frames
                self.delayed = deque((old, True) for old, _ in self.delayed)
                voiced = True
                self.in_utterance = True
                self.speech_frames += 1
            else:
                voiced = self.hold_frames > 0
                self.hold_frames = max(0, self.hold_frames - 1)
                if self.in_utterance and self.hold_frames == 0:
                    if self.speech_frames >= 6:
                        self.speech_end_events += 1
                    self.in_utterance = False
                    self.speech_frames = 0
            self.delayed.append((frame, voiced))
            if len(self.delayed) > self.pre_roll_frames:
                old_frame, keep = self.delayed.popleft()
                output.append(old_frame if keep else np.zeros_like(old_frame))
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)
