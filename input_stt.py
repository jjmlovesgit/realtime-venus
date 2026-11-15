"""Independent local transcript of microphone audio for debugging duplex sessions."""

import json
import queue
import threading


MIN_PHRASE_RMS = 10 ** (-45 / 20)


class InputTranscriber:
    def __init__(self, model_path, sample_rate, np):
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.np = np
        self.recognizer = KaldiRecognizer(Model(str(model_path)), sample_rate)
        self.blocks = queue.Queue(maxsize=100)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="venus-input-stt", daemon=True)
        self.dropped_blocks = 0

    def start(self):
        self.thread.start()

    def submit(self, samples):
        try:
            self.blocks.put_nowait(samples)
        except queue.Full:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                pass
            self.blocks.put_nowait(samples)
            self.dropped_blocks += 1

    def _emit(self, result, had_voice):
        text = json.loads(result).get("text", "").strip()
        if text and had_voice:
            print(f"\n[input STT] {text}", flush=True)

    def _run(self):
        phrase_has_voice = False
        while not self.stop.is_set() or not self.blocks.empty():
            try:
                samples = self.blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            block_rms = float(self.np.sqrt(self.np.mean(samples * samples))) if len(samples) else 0.0
            phrase_has_voice |= block_rms >= MIN_PHRASE_RMS
            pcm = (self.np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            if self.recognizer.AcceptWaveform(pcm):
                self._emit(self.recognizer.Result(), phrase_has_voice)
                phrase_has_voice = False
        self._emit(self.recognizer.FinalResult(), phrase_has_voice)
        if self.dropped_blocks:
            print(f"[input STT] dropped {self.dropped_blocks} transcription blocks", flush=True)

    def close(self):
        self.stop.set()
        self.thread.join()
