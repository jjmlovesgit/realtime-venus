"""Quiet, seamless radio bed mixed only into unused speaker frames."""

from math import gcd
import threading

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class IdleAudioLoop:
    def __init__(self, samples, sample_rate, level_dbfs=-48.0):
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(source) < sample_rate // 4:
            raise ValueError("Idle audio must be at least 0.25 seconds long")
        # MP3 encoder padding and the source's opening/closing silence otherwise
        # become a repeated gap on every loop. Only trim the outer edges.
        frame_samples = max(1, sample_rate // 100)
        frame_rms = np.asarray([
            np.sqrt(np.mean(frame * frame))
            for frame in (source[pos:pos + frame_samples]
                          for pos in range(0, len(source), frame_samples))
        ])
        threshold = max(1e-5, float(np.median(frame_rms)) * 0.18)
        active = np.flatnonzero(frame_rms >= threshold)
        if not len(active):
            raise ValueError("Idle audio contains no audible signal")
        start = int(active[0] * frame_samples)
        end = int(min(len(source), (active[-1] + 1) * frame_samples))
        self.trimmed_leading_seconds = start / sample_rate
        self.trimmed_trailing_seconds = (len(source) - end) / sample_rate
        source = source[start:end]
        rms = float(np.sqrt(np.mean(source * source)))
        if rms < 1e-6:
            raise ValueError("Idle audio contains no audible signal")

        # Equal-power overlap avoids a volume dip for uncorrelated radio noise.
        overlap = min(round(sample_rate * 0.06), len(source) // 6)
        phase = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
        seam = source[-overlap:] * np.cos(phase) + source[:overlap] * np.sin(phase)
        loop = np.concatenate((source[overlap:-overlap], seam))

        target_rms = 10.0 ** (level_dbfs / 20.0)
        gain = min(target_rms / rms, 0.08 / float(np.max(np.abs(loop))))
        self.samples = np.asarray(loop * gain, dtype=np.float32)
        self.crossfade_seconds = overlap / sample_rate
        self.sample_rate = sample_rate
        self.fade_samples = max(1, round(sample_rate * 0.025))
        self.position = 0
        self.gain = 0.0

    @classmethod
    def from_file(cls, path, sample_rate, level_dbfs=-48.0):
        channels, source_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = channels.mean(axis=1, dtype=np.float32)
        if source_rate != sample_rate:
            divisor = gcd(source_rate, sample_rate)
            mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
        return cls(mono, sample_rate, level_dbfs)

    def mix_into(self, output, idle_mask):
        """Add static where the speaker queue is empty, fading at transitions."""
        count = len(output)
        if not count:
            return
        mask = np.broadcast_to(np.asarray(idle_mask, dtype=np.bool_), (count,))
        offsets = (self.position + np.arange(count)) % len(self.samples)
        bed = self.samples[offsets]
        self.position = (self.position + count) % len(self.samples)

        gains = np.empty(count, dtype=np.float32)
        edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [count]))
        for start, end in zip(starts, ends):
            target = float(mask[start])
            decay = np.exp(-np.arange(1, end - start + 1) / self.fade_samples)
            segment = target + (self.gain - target) * decay
            gains[start:end] = segment
            self.gain = float(segment[-1])
        output += bed * gains


class UIIdlePlayer:
    """Play the same quiet bed while the app has no live model session."""

    SAMPLE_RATE = 48_000

    def __init__(self, path, on_error=None):
        self.path = path
        self.on_error = on_error
        self.thread = None
        self.stop_event = None
        self.settings = None

    def start(self, device=None, level_dbfs=-48.0):
        settings = (device, level_dbfs)
        if self.thread is not None and self.thread.is_alive() and self.settings == settings:
            return
        self.stop()
        self.settings = settings
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, args=(self.stop_event, device, level_dbfs),
            name="venus-idle-radio", daemon=True,
        )
        self.thread.start()

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        self.settings = None

    def _run(self, stop_event, device, level_dbfs):
        try:
            import sounddevice as sd

            loop = IdleAudioLoop.from_file(self.path, self.SAMPLE_RATE, level_dbfs)
            block = self.SAMPLE_RATE // 10
            with sd.OutputStream(
                samplerate=self.SAMPLE_RATE, blocksize=block,
                device=device, channels=1, dtype="float32",
            ) as output:
                while not stop_event.is_set():
                    frame = np.zeros(block, dtype=np.float32)
                    loop.mix_into(frame, True)
                    output.write(frame)
                frame = np.zeros(block, dtype=np.float32)
                loop.mix_into(frame, False)
                output.write(frame)
        except Exception as exc:
            if not stop_event.is_set() and self.on_error is not None:
                self.on_error(exc)
