#!/usr/bin/env python3
"""Talk to Realtime-Venus-Audio through a live microphone and speakers."""

import argparse
from collections import deque
from contextlib import nullcontext
import json
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
DEFAULT_PLAYBACK_TAIL_SECONDS = 0.5
CAPTURE_BLOCK_SAMPLES = 1_600  # 100 ms
SENTENCE_END_RE = re.compile(r"[.!?…。？！][\"'”’)]?\s*$")
ENGLISH_ONLY_INSTRUCTIONS = (
    "You are an English-language voice assistant. All replies and spoken output "
    "must be in English. Do not answer in Chinese or switch languages unless the "
    "user explicitly asks you to speak Chinese. If the user speaks another language, "
    "answer in English."
)
ANSWER_INSTEAD_OF_ECHO_INSTRUCTIONS = (
    "Treat microphone speech as the user's request and answer it directly. "
    "Do not transcribe, repeat, or imitate the user's words unless they explicitly "
    "ask you to repeat them. If they ask for a story, begin the actual story "
    "in your first sentence and finish it in this turn. Never announce that "
    "you will tell a story or wait for 'go ahead'."
)
WEATHER_INSTRUCTIONS = (
    "The weather tool is optional. Use it only when the latest spoken request explicitly "
    "asks for current weather in a five-digit U.S. ZIP code. Then emit a single "
    "delegate JSON object with tool weather_by_zip and zip_code set to the spoken ZIP. "
    "For every other request, answer aloud normally without a delegate."
)
CALCULATOR_INSTRUCTIONS = (
    "The calculator is optional. Use it only when the latest spoken request explicitly "
    "asks for arithmetic. Then emit a single delegate JSON object with tool calculator "
    "and expression set to the spoken arithmetic. For every other request, "
    "answer aloud normally without a delegate."
)
TIMER_INSTRUCTIONS = (
    "Never "
    "start or cancel a timer unless the user explicitly asks for that action in the "
    "current turn. Counting, arithmetic, or merely mentioning a number or time is "
    "not a timer request. For timer_start, include user_request containing the exact "
    "short phrase the user spoke, and use duration_seconds as a number, for example "
    '<delegate>{"tool":"timer_start","duration_seconds":120,"label":"tea",'
    '"user_request":"Set a timer for two minutes"}</delegate>. If the user did not '
    "clearly ask to start a timer, ask what they want; do not call the tool. For "
    "timer_cancel, include the user's explicit cancellation phrase in user_request. "
    "When a timer-expired result arrives, announce its short alert. Never claim a "
    "timer was started or canceled unless the tool result confirms it."
)


def build_legacy_system_prompt(user_prompt, calculator=False, timer=False, weather=False):
    """Recreate the prompt used before the UI exposed the complete prompt."""
    user_prompt = user_prompt.strip()
    prefix = "Streaming Omni Conversation."
    if user_prompt.startswith(prefix):
        user_prompt = user_prompt[len(prefix):].strip()
    parts = [prefix, user_prompt, ENGLISH_ONLY_INSTRUCTIONS,
             ANSWER_INSTEAD_OF_ECHO_INSTRUCTIONS]
    if calculator:
        parts.append(CALCULATOR_INSTRUCTIONS)
    if timer:
        parts.append(TIMER_INSTRUCTIONS)
    if weather:
        parts.append(WEATHER_INSTRUCTIONS)
    return " ".join(part for part in parts if part)


def resolve_system_prompt(args):
    """The UI sends exact text; older CLI calls retain their prior defaults."""
    if args.exact_system_prompt:
        return args.system_prompt
    return build_legacy_system_prompt(
        args.system_prompt, args.calculator_tool, args.timer_tool, args.weather_tool,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Realtime-Venus-Audio")
    parser.add_argument("--input-device", type=int, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, help="sounddevice output device index")
    parser.add_argument(
        "--voice-reference",
        help="Optional WAV file for the generated voice (default: checkpoint's HT_ref_audio.wav)",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--sentence-buffering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for punctuation or turn end before playing a sentence (default: play chunks immediately)",
    )
    parser.add_argument(
        "--weather-tool",
        action="store_true",
        help="Enable the experimental ZIP-code weather tool",
    )
    parser.add_argument(
        "--calculator-tool",
        action="store_true",
        help="Enable the local arithmetic calculator",
    )
    parser.add_argument(
        "--timer-tool",
        action="store_true",
        help="Enable local countdown timers",
    )
    parser.add_argument("--system-prompt", default="Streaming Omni Conversation.")
    parser.add_argument(
        "--exact-system-prompt", action="store_true",
        help="Send --system-prompt verbatim without built-in prefix or rules",
    )
    parser.add_argument("--decode-mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--max-new-tokens-per-chunk", type=int, default=40)
    parser.add_argument("--tts-max-new-tokens-per-chunk", type=int, default=64)
    parser.add_argument("--tts-final-min-tokens", type=int, default=32)
    parser.add_argument("--tts-final-max-tokens", type=int, default=128)
    parser.add_argument("--playback-tail-seconds", type=float, default=DEFAULT_PLAYBACK_TAIL_SECONDS)
    parser.add_argument(
        "--idle-static", action=argparse.BooleanOptionalAction, default=True,
        help="Loop a quiet radio bed whenever the speaker queue is idle",
    )
    parser.add_argument("--idle-static-level-dbfs", type=float, default=-48.0)
    parser.add_argument("--mic-gain-db", type=float, default=-2.0)
    parser.add_argument("--speaker-queue-size", type=int, default=8)
    parser.add_argument(
        "--mute-mic-during-playback", action=argparse.BooleanOptionalAction, default=False,
        help="Feed silence to Venus while its reply plays; disables voice interruption",
    )
    parser.add_argument(
        "--echo-cancellation", action=argparse.BooleanOptionalAction, default=False,
        help="Use WebRTC AEC3 with a duplex speaker reference; keeps the mic live",
    )
    parser.add_argument(
        "--silero-vad", action=argparse.BooleanOptionalAction, default=False,
        help="Filter microphone non-speech with local Silero VAD",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.6)
    parser.add_argument(
        "--save-last-reply-debug", action="store_true",
        help="Save each generated reply as logs/reply-*.wav for diagnosis",
    )
    parser.add_argument(
        "--input-stt", action=argparse.BooleanOptionalAction, default=True,
        help="Log a separate local English transcription of the microphone (default: on)",
    )
    parser.add_argument(
        "--stt-model-path", default="models/vosk-model-small-en-us-0.15",
        help="Local Vosk model directory for microphone transcription",
    )
    parser.add_argument("--control-stdin", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument(
        "--reply-after-speech", action=argparse.BooleanOptionalAction, default=True,
        help="Briefly favor a reply after speech ends instead of staying in listen",
    )
    parser.add_argument(
        "--keep-last-turns", type=int, default=0,
        help="Keep this many completed user/Venus exchanges in context (0 disables trimming)",
    )
    return parser.parse_args()


def mono_samples(waveform, np):
    """Accept the sample/channel layouts used by the upstream WAV writer."""
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(f"Unexpected generated audio shape: {samples.shape}")
    if samples.shape[0] <= 8:
        samples = samples.T
    return samples.mean(axis=1, dtype=np.float32)


def scope_frame(samples, np):
    """Compact speaker waveform telemetry for the UI; never written to session logs."""
    if not len(samples):
        return None
    signal = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(signal * signal)))
    if rms < 0.001:
        return None
    peaks = []
    for segment in np.array_split(signal, 48):
        peaks.append(int(round(float(segment[np.argmax(np.abs(segment))]) * 100))
                     if segment.size else 0)
    return f"[scope] {rms:.4f} " + ",".join(map(str, peaks))


def load_playback_cue(path, np):
    """Decode a cue once as mono 24 kHz playback samples."""
    import soundfile as sf
    from scipy.signal import resample_poly

    channels, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    samples = channels.mean(axis=1, dtype=np.float32)
    if sample_rate != OUTPUT_SAMPLE_RATE:
        samples = resample_poly(samples, OUTPUT_SAMPLE_RATE, sample_rate)
    return np.asarray(samples, dtype=np.float32)


def queue_reply_ending(queue_audio, cue, tail):
    """Play a separate cue after speech, then the normal end-of-turn tail."""
    if cue is not None and cue.size:
        queue_audio(cue, include_in_debug=False)
    queue_audio(tail, end_turn=True)


def put_latest(items, value):
    """Keep microphone latency bounded if inference falls behind."""
    try:
        items.put_nowait(value)
        return False
    except queue.Full:
        try:
            items.get_nowait()
        except queue.Empty:
            pass
        items.put_nowait(value)
        return True


def use_soundfile_for_reference_wav(torch):
    """Avoid TorchCodec's FFmpeg DLL dependency for the bundled WAV prompt."""
    import soundfile as sf
    import torchaudio

    original_load = torchaudio.load

    def load_audio(filepath, *args, **kwargs):
        if str(filepath).lower().endswith(".wav") and not args and set(kwargs) <= {"backend"}:
            samples, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
            return torch.from_numpy(samples.T.copy()), sample_rate
        return original_load(filepath, *args, **kwargs)

    torchaudio.load = load_audio


def trim_synthetic_leading_silence(samples, np):
    """Remove exact zero padding added before short non-final TTS chunks."""
    audible = np.flatnonzero(samples)
    if audible.size:
        return samples[audible[0]:]
    return samples[:0]


def ensure_tts_token_control(model_path):
    """Restore local TTS controls after a checkpoint download replaces model code."""
    source = model_path / "modeling_minicpmo.py"
    code = source.read_text(encoding="utf-8")
    replacements = []
    if "tts_max_new_tokens_per_chunk=26" not in code:
        replacements.extend((
            (
                "        max_new_speak_tokens_per_chunk=20,\n        decode_mode: str =",
                "        max_new_speak_tokens_per_chunk=20,\n"
                "        tts_max_new_tokens_per_chunk=26,\n        decode_mode: str =",
            ),
            (
                "        max_token_per_chunk = 25 + 1\n"
                "        min_token_per_chunk = 25 + 1",
                "        max_token_per_chunk = tts_max_new_tokens_per_chunk\n"
                "        min_token_per_chunk = min(25 + 1, max_token_per_chunk)",
            ),
        ))
    if "tts_final_min_tokens=0" not in code:
        replacements.extend((
            (
                "        tts_max_new_tokens_per_chunk=26,\n        decode_mode: str =",
                "        tts_max_new_tokens_per_chunk=26,\n"
                "        tts_final_min_tokens=0,\n        decode_mode: str =",
            ),
            (
                "        if end_of_turn:\n            min_token_per_chunk = 0\n"
                "        force_flush = False",
                "        final_tts_padding = bool(\n"
                "            end_of_turn and not text.strip() and self.tts_text_start_pos > 0\n"
                "            and tts_final_min_tokens > 0\n"
                "        )\n"
                "        if end_of_turn:\n"
                "            min_token_per_chunk = min(tts_final_min_tokens, max_token_per_chunk - 1) if final_tts_padding else 0\n"
                "        force_flush = False",
            ),
            (
                '            "n_tts_tokens": new_tokens.numel(),\n',
                '            "n_tts_tokens": new_tokens.numel(),\n'
                '            "tts_final_min_applied": min_token_per_chunk if final_tts_padding else 0,\n',
            ),
        ))
    if "tts_final_max_tokens=128" not in code:
        replacements.extend((
            (
                "        tts_final_min_tokens=0,\n        decode_mode: str =",
                "        tts_final_min_tokens=0,\n"
                "        tts_final_max_tokens=128,\n        decode_mode: str =",
            ),
            (
                "        max_token_per_chunk = tts_max_new_tokens_per_chunk\n",
                "        max_token_per_chunk = tts_final_max_tokens if end_of_turn else tts_max_new_tokens_per_chunk\n",
            ),
            (
                '            "n_tts_tokens": new_tokens.numel(),\n',
                '            "n_tts_tokens": new_tokens.numel(),\n'
                '            "tts_token_limit": max_token_per_chunk,\n',
            ),
        ))
    if not replacements:
        return
    for old, new in replacements:
        if code.count(old) != 1:
            raise RuntimeError(
                f"Unsupported checkpoint model code in {source}; "
                "cannot apply the TTS token control safely"
            )
        code = code.replace(old, new, 1)
    source.write_text(code, encoding="utf-8")
    print("Applied local TTS token control to checkpoint model code.", flush=True)


def run(args):
    import numpy as np
    import sounddevice as sd
    if args.save_last_reply_debug:
        import soundfile as sf

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.max_new_tokens_per_chunk < 2:
        raise ValueError("--max-new-tokens-per-chunk must be at least 2")
    if not 26 <= args.tts_max_new_tokens_per_chunk <= 128:
        raise ValueError("--tts-max-new-tokens-per-chunk must be between 26 and 128")
    if not 26 <= args.tts_final_max_tokens <= 256:
        raise ValueError("--tts-final-max-tokens must be between 26 and 256")
    if not 0 <= args.tts_final_min_tokens < args.tts_final_max_tokens:
        raise ValueError("--tts-final-min-tokens must be below the final TTS token limit")
    if not 0 <= args.playback_tail_seconds <= 2:
        raise ValueError("--playback-tail-seconds must be between 0 and 2")
    if not -65 <= args.idle_static_level_dbfs <= -25:
        raise ValueError("--idle-static-level-dbfs must be between -65 and -25")
    if not -24 <= args.mic_gain_db <= 12:
        raise ValueError("--mic-gain-db must be between -24 and 12")
    if not 1 <= args.speaker_queue_size <= 64:
        raise ValueError("--speaker-queue-size must be between 1 and 64")
    if not 0.1 <= args.vad_threshold <= 0.95:
        raise ValueError("--vad-threshold must be between 0.1 and 0.95")
    if not 0 <= args.keep_last_turns <= 50:
        raise ValueError("--keep-last-turns must be between 0 and 50")

    import torch
    from transformers import AutoModel, set_seed
    from live_tools import DelegateParser, TimerManager, execute_local_tool
    from response_nudge import ResponseNudge
    from context_retention import TurnRetention, DEFAULT_HIGH_TOKENS, DEFAULT_LOW_TOKENS
    mic_gain = 10.0 ** (args.mic_gain_db / 20.0)
    if args.weather_tool:
        from weather_tool import run_weather_delegate
    else:
        run_weather_delegate = None

    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {model_path / 'config.json'}")
    ensure_tts_token_control(model_path)
    project_reference = Path(__file__).resolve().parent / "assets" / "voice_reference.wav"
    default_reference = project_reference if project_reference.is_file() else model_path / "assets" / "HT_ref_audio.wav"
    reference_audio = (
        Path(args.voice_reference).expanduser().resolve()
        if args.voice_reference
        else default_reference
    )
    if not reference_audio.is_file():
        raise FileNotFoundError(f"TTS reference audio not found: {reference_audio}")
    if reference_audio.suffix.lower() != ".wav":
        raise ValueError("--voice-reference must point to a WAV file")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")

    cue_path = Path(__file__).resolve().parent / "assets" / "mic-off.mp3"
    shared_cue = None
    if cue_path.is_file():
        try:
            shared_cue = load_playback_cue(cue_path, np)
            print(
                f"[audio cue] mic in/out sound loaded once "
                f"({len(shared_cue) / OUTPUT_SAMPLE_RATE:.2f}s)",
                flush=True,
            )
        except (OSError, ValueError) as exc:
            print(f"[audio cue] could not load {cue_path.name}: {exc}", file=sys.stderr, flush=True)
    reply_end_cue = shared_cue
    mic_input_cue = shared_cue

    idle_loop = None
    if args.idle_static:
        idle_path = Path(__file__).resolve().parent / "assets" / "radio-static-idle.mp3"
        if idle_path.is_file():
            try:
                from idle_audio import IdleAudioLoop
                idle_loop = IdleAudioLoop.from_file(
                    idle_path, 48_000 if args.echo_cancellation else OUTPUT_SAMPLE_RATE,
                    args.idle_static_level_dbfs,
                )
                print(
                    f"[idle audio] radio static at {args.idle_static_level_dbfs:.0f} dBFS; "
                    f"trimmed {idle_loop.trimmed_leading_seconds:.2f}s start, "
                    f"{idle_loop.trimmed_trailing_seconds:.2f}s end; "
                    f"loop crossfade {idle_loop.crossfade_seconds * 1000:.0f}ms",
                    flush=True,
                )
            except (OSError, ValueError) as exc:
                print(f"[idle audio] could not load {idle_path.name}: {exc}", file=sys.stderr, flush=True)

    io_rate = 48_000 if args.echo_cancellation else INPUT_SAMPLE_RATE
    sd.check_input_settings(device=args.input_device, channels=1, samplerate=io_rate)
    sd.check_output_settings(
        device=args.output_device, channels=1,
        samplerate=48_000 if args.echo_cancellation else OUTPUT_SAMPLE_RATE,
    )

    set_seed(42)
    print(f"Loading {model_path} on {torch.cuda.get_device_name(0)} ...", flush=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
    )
    model.eval().cuda()
    if args.keep_last_turns and int(model.config.max_position_embeddings) <= DEFAULT_HIGH_TOKENS:
        raise ValueError("Checkpoint context limit is too small for the turn-retention safety cap")
    duplex = model.as_duplex(generate_audio=True)
    enabled_tools = set()
    if args.weather_tool:
        enabled_tools.add("weather_by_zip")
    if args.calculator_tool:
        enabled_tools.add("calculator")
    if args.timer_tool:
        enabled_tools.update({"timer_start", "timer_cancel"})
    delegate_parser = DelegateParser(duplex.tokenizer) if enabled_tools else None
    use_soundfile_for_reference_wav(torch)
    system_prompt = resolve_system_prompt(args)
    print(f"Effective system prompt: {system_prompt}", flush=True)
    duplex.prepare(prefix_system_prompt=system_prompt, prompt_wav_path=str(reference_audio))
    turn_retention = TurnRetention(args.keep_last_turns) if args.keep_last_turns else None
    if turn_retention is not None:
        initial_cache = duplex.decoder.get_cache_length()
        if initial_cache >= DEFAULT_LOW_TOKENS:
            raise ValueError("System prompt is too large to preserve with the context safety cap")
        print(
            f"[context] keep_last_turns={args.keep_last_turns} "
            f"safety={DEFAULT_HIGH_TOKENS}/{DEFAULT_LOW_TOKENS} "
            f"prompt_cache={initial_cache}",
            flush=True,
        )
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.1f} GiB", flush=True)

    if duplex.SAMPLE_RATE != INPUT_SAMPLE_RATE:
        raise RuntimeError(f"Expected {INPUT_SAMPLE_RATE} Hz input; model requires {duplex.SAMPLE_RATE} Hz")
    chunk_samples = round(duplex.CHUNK_MS * duplex.SAMPLE_RATE / 1000)
    if chunk_samples <= 0:
        raise RuntimeError("Model reported an invalid chunk size")

    mic_blocks = queue.Queue(maxsize=100)
    speaker_chunks = queue.Queue(maxsize=args.speaker_queue_size)
    speaker_errors = queue.Queue(maxsize=1)
    scope_frames = queue.Queue(maxsize=4)
    speaker_gate = threading.Event()
    debug_audio_chunks = []
    debug_audio_dir = Path(__file__).resolve().parent / "logs"
    debug_session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    debug_reply_index = 0
    stop = threading.Event()
    request_stop = threading.Event()
    timer_events = queue.Queue()
    timers = TimerManager(timer_events)
    dropped_mic_blocks = 0
    transcriber = None
    vad_filter = None
    if args.silero_vad:
        try:
            from mic_vad import SileroMicFilter
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is missing. Install it with "
                "python -m pip install --no-deps -r requirements-vad.txt"
            ) from exc
        vad_filter = SileroMicFilter(threshold=args.vad_threshold)
        print(f"[VAD] Silero ONNX CPU enabled; threshold={args.vad_threshold:.2f}", flush=True)
    mic_cue_detector = vad_filter
    if mic_input_cue is not None and mic_cue_detector is None:
        try:
            from mic_vad import SileroMicFilter
            mic_cue_detector = SileroMicFilter(threshold=args.vad_threshold)
            print("[mic cue] Silero detects speech endings without filtering mic input", flush=True)
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"[mic cue] speech-end detector unavailable: {exc}", file=sys.stderr, flush=True)
            mic_input_cue = None
    if args.input_stt:
        from input_stt import InputTranscriber

        stt_model = Path(args.stt_model_path).expanduser().resolve()
        if not (stt_model / "am" / "final.mdl").is_file():
            raise FileNotFoundError(
                f"STT model not found at {stt_model}. Run python download_stt_model.py "
                "or use --no-input-stt."
            )
        transcriber = InputTranscriber(stt_model, INPUT_SAMPLE_RATE, np)
        print("[input STT] Local diagnostic transcript enabled (Vosk).", flush=True)

    aec_io = None
    if args.echo_cancellation:
        try:
            from duplex_aec import DuplexAecIO, STREAM_BLOCK, STREAM_RATE
        except ImportError as exc:
            raise RuntimeError(
                "WebRTC AEC is missing. Install with "
                "python -m pip install --no-deps -r requirements-aec.txt"
            ) from exc
        aec_io = DuplexAecIO(
            mic_blocks, speaker_chunks, speaker_errors,
            transcriber if vad_filter is None else None,
            put_latest, speaker_gate, args.mute_mic_during_playback, scope_frames,
            idle_loop, mic_gain,
        )
        print("[AEC] WebRTC AEC3 enabled; mic remains live for interruption.", flush=True)

    if args.control_stdin:
        def read_control():
            for line in sys.stdin:
                if line.strip().lower() == "stop":
                    request_stop.set()
                    return

        threading.Thread(target=read_control, name="venus-ui-control", daemon=True).start()

    def mic_callback(indata, frames, time_info, status):
        nonlocal dropped_mic_blocks
        if status:
            print(f"[microphone] {status}", file=sys.stderr, flush=True)
        samples = indata[:, 0].copy()
        np.multiply(samples, mic_gain, out=samples)
        np.clip(samples, -1.0, 1.0, out=samples)
        if args.mute_mic_during_playback and speaker_gate.is_set():
            samples.fill(0)
        if put_latest(mic_blocks, samples):
            dropped_mic_blocks += 1
        if transcriber is not None and vad_filter is None:
            transcriber.submit(samples)

    def speaker_worker():
        try:
            if aec_io is not None:
                with sd.Stream(
                    samplerate=STREAM_RATE,
                    blocksize=STREAM_BLOCK,
                    device=(args.input_device, args.output_device),
                    channels=(1, 1),
                    dtype="float32",
                    callback=aec_io.callback,
                ) as stream:
                    aec_io.input_latency = stream.latency[0]
                    print(
                        f"[AEC] input latency={aec_io.input_latency * 1000:.0f}ms; "
                        "using stream clock if capture timestamps are unavailable",
                        flush=True,
                    )
                    pending_turns = deque()
                    while (not stop.is_set() or not speaker_chunks.empty() or
                           aec_io.playing or pending_turns or not aec_io.completions.empty()):
                        if not speaker_errors.empty():
                            return
                        latest_frame = None
                        while True:
                            try:
                                latest_frame = scope_frames.get_nowait()
                            except queue.Empty:
                                break
                        if latest_frame is not None:
                            telemetry = scope_frame(latest_frame, np)
                            if telemetry is not None:
                                print(telemetry, flush=True)
                        try:
                            kind, value = aec_io.completions.get(timeout=0.05)
                        except queue.Empty:
                            kind = None
                        if kind == "status":
                            print(f"[audio stream] {value}", file=sys.stderr, flush=True)
                        elif kind == "turn":
                            pending_turns.append(value)
                        while pending_turns and stream.time >= pending_turns[0] + 0.2:
                            pending_turns.popleft()
                            print("[playback complete] reply reached speaker DAC", flush=True)
                            if (args.mute_mic_during_playback and speaker_chunks.empty()
                                    and not aec_io.playing):
                                speaker_gate.clear()
                return
            with sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                device=args.output_device,
                channels=1,
                dtype="float32",
            ) as output:
                while not stop.is_set() or not speaker_chunks.empty():
                    try:
                        if idle_loop is None:
                            samples, end_turn = speaker_chunks.get(timeout=0.1)
                        else:
                            samples, end_turn = speaker_chunks.get_nowait()
                    except queue.Empty:
                        if idle_loop is not None:
                            frame = np.zeros(OUTPUT_SAMPLE_RATE // 10, dtype=np.float32)
                            idle_loop.mix_into(frame, True)
                            output.write(frame)
                        continue
                    for offset in range(0, len(samples), OUTPUT_SAMPLE_RATE // 10):
                        speech_frame = samples[offset:offset + OUTPUT_SAMPLE_RATE // 10]
                        frame = speech_frame.copy() if idle_loop is not None else speech_frame
                        if idle_loop is not None:
                            idle_loop.mix_into(frame, False)
                        if output.write(frame):
                            print("[playback] speaker underflow", file=sys.stderr, flush=True)
                        telemetry = scope_frame(speech_frame, np)
                        if telemetry is not None:
                            print(telemetry, flush=True)
                    if end_turn:
                        print("[playback complete] queued reply written to speaker", flush=True)
                    if end_turn and args.mute_mic_during_playback:
                        time.sleep(0.3)
                        if speaker_chunks.empty():
                            speaker_gate.clear()
                            print("[input gate] microphone resumed", flush=True)
        except Exception as exc:
            if speaker_errors.empty():
                speaker_errors.put_nowait(exc)

    def queue_speaker_audio(samples, end_turn=False, include_in_debug=True):
        """Preserve queued speech even when playback trails generation."""
        nonlocal debug_reply_index
        if args.save_last_reply_debug:
            if include_in_debug:
                debug_audio_chunks.append(samples.copy())
            if end_turn:
                debug_reply_index += 1
                debug_audio_path = debug_audio_dir / (
                    f"reply-{debug_session_id}-{debug_reply_index:03}.wav"
                )
                debug_audio_dir.mkdir(exist_ok=True)
                sf.write(
                    debug_audio_path, np.concatenate(debug_audio_chunks),
                    OUTPUT_SAMPLE_RATE, subtype="PCM_16",
                )
                debug_audio_chunks.clear()
                print(f"[audio diagnostic] generated reply: {debug_audio_path}", flush=True)
        if args.mute_mic_during_playback and not speaker_gate.is_set():
            speaker_gate.set()
            print("[input gate] microphone muted during Venus playback", flush=True)
        playback_samples = aec_io.speaker_samples(samples) if aec_io is not None else samples
        while True:
            if not speaker_errors.empty():
                raise RuntimeError("Speaker playback failed") from speaker_errors.get_nowait()
            try:
                speaker_chunks.put((playback_samples, end_turn), timeout=0.1)
                return
            except queue.Full:
                continue

    speaker = threading.Thread(target=speaker_worker, name="venus-speaker", daemon=True)
    speaker.start()
    if transcriber is not None:
        transcriber.start()
    pending = np.empty(0, dtype=np.float32)
    previous_speech_audio = False
    chunk_index = 0
    last_aec_report_chunk = 0
    pending_backend_texts = []
    sentence_audio_chunks = []
    sentence_text = ""
    turn_text_parts = []
    turn_audio_seconds = 0.0
    quiet_chunks = 0
    response_nudge = ResponseNudge() if args.reply_after_speech and vad_filter is not None else None

    def flush_sentence():
        nonlocal sentence_text
        if sentence_audio_chunks:
            sentence_audio = np.concatenate(sentence_audio_chunks)
            if sentence_audio.size:
                queue_speaker_audio(sentence_audio)
        if sentence_text.strip():
            print(f"\n[spoken sentence] {sentence_text.strip()}", flush=True)
        sentence_audio_chunks.clear()
        sentence_text = ""

    print(
        "Playback: sentence buffering" if args.sentence_buffering else "Playback: immediate chunks",
        flush=True,
    )
    print(
        f"Active playback settings: text_tokens={args.max_new_tokens_per_chunk} "
        f"tts_tokens={args.tts_max_new_tokens_per_chunk} "
        f"final_tts_min={args.tts_final_min_tokens} "
        f"final_tts_max={args.tts_final_max_tokens} "
        f"tail={args.playback_tail_seconds:.2f}s "
        f"speaker_queue={args.speaker_queue_size} "
        f"mute_mic_during_playback={args.mute_mic_during_playback} "
        f"echo_cancellation={args.echo_cancellation} "
        f"reply_after_speech={args.reply_after_speech} "
        f"keep_last_turns={args.keep_last_turns} "
        f"mic_gain_db={args.mic_gain_db:.1f} "
        f"silero_vad={args.silero_vad} vad_threshold={args.vad_threshold:.2f}",
        flush=True,
    )
    if not args.mute_mic_during_playback and not args.echo_cancellation:
        print(
            "[playback] Mic is live without echo cancellation; speaker sound "
            "may be mistaken for a user turn.",
            flush=True,
        )
    if args.weather_tool or args.calculator_tool or args.timer_tool:
        print(f"Enabled tools: {', '.join(sorted(enabled_tools))}.", flush=True)
    if args.weather_tool:
        print("Weather lookup needs internet; calculator and timers run locally.", flush=True)
    print("Listening. Press Ctrl+C to stop. Use headphones for best echo rejection.", flush=True)
    try:
        input_stream = nullcontext() if aec_io is not None else sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            device=args.input_device,
            channels=1,
            dtype="float32",
            blocksize=CAPTURE_BLOCK_SAMPLES,
            callback=mic_callback,
        )
        with input_stream:
            with torch.inference_mode():
                while not request_stop.is_set():
                    if not speaker_errors.empty():
                        raise RuntimeError("Speaker playback failed") from speaker_errors.get_nowait()
                    while True:
                        try:
                            pending_backend_texts.append(
                                f"<backend>{timer_events.get_nowait()}</backend>"
                            )
                        except queue.Empty:
                            break
                    try:
                        block = mic_blocks.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if vad_filter is not None:
                        block = vad_filter.process(block)
                        if not block.size:
                            continue
                        if transcriber is not None:
                            transcriber.submit(block)
                    elif mic_cue_detector is not None:
                        mic_cue_detector.process(block)
                    if (mic_cue_detector is not None and mic_input_cue is not None
                            and mic_cue_detector.consume_speech_ends()):
                        queue_speaker_audio(mic_input_cue, include_in_debug=False)
                    pending = np.concatenate((pending, block))
                    while len(pending) >= chunk_samples:
                        chunk_index += 1
                        chunk = pending[:chunk_samples].copy()
                        pending = pending[chunk_samples:]
                        mic_rms = float(np.sqrt(np.mean(chunk * chunk)))
                        mic_dbfs = 20.0 * np.log10(max(mic_rms, 1e-8))
                        generate_started = time.monotonic()
                        prefill_text = pending_backend_texts or None
                        pending_backend_texts = []
                        prefill = duplex.streaming_prefill(
                            audio_waveform=chunk,
                            text_list=prefill_text,
                        )
                        if not prefill.get("success"):
                            raise RuntimeError(prefill.get("reason") or "Audio prefill failed")
                        nudge_attempt = 0
                        if response_nudge is not None:
                            nudge_attempt = response_nudge.before_generate(
                                mic_dbfs, duplex.pending_logits, duplex.listen_token_id,
                            )
                            if nudge_attempt:
                                print(
                                    f"[response nudge] attempt {nudge_attempt}/2 "
                                    "after detected speech",
                                    flush=True,
                                )
                        generated_start = len(duplex.total_ids)
                        result = duplex.streaming_generate(
                            max_new_speak_tokens_per_chunk=args.max_new_tokens_per_chunk,
                            tts_max_new_tokens_per_chunk=args.tts_max_new_tokens_per_chunk,
                            tts_final_min_tokens=args.tts_final_min_tokens,
                            tts_final_max_tokens=args.tts_final_max_tokens,
                            decode_mode=args.decode_mode,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            listen_prob_scale=args.listen_prob_scale,
                        )
                        if delegate_parser is not None:
                            delegate_touched, _, delegate_body = delegate_parser.consume(
                                duplex.total_ids[generated_start:]
                            )
                        else:
                            delegate_touched, delegate_body = False, None
                        inference_seconds = time.monotonic() - generate_started
                        text_out = result.get("text", "")
                        is_listen = result.get("is_listen")
                        if response_nudge is not None and response_nudge.after_generate(is_listen):
                            print("[response nudge] model kept listening", flush=True)
                        end_of_turn = bool(result.get("end_of_turn"))
                        audio_seconds = 0.0
                        tts_limit = result.get("tts_token_limit", args.tts_max_new_tokens_per_chunk)
                        if end_of_turn and result.get("n_tts_tokens", 0) >= tts_limit - 1:
                            print(
                                "[playback] Final TTS may have hit its token limit; "
                                "raise Final TTS max if speech is missing",
                                file=sys.stderr,
                                flush=True,
                            )
                        if (
                            result.get("audio_waveform") is not None
                            and not is_listen
                            and not delegate_touched
                        ):
                            samples = mono_samples(result["audio_waveform"], np)
                            if args.sentence_buffering:
                                samples = trim_synthetic_leading_silence(samples, np)
                            audio_seconds = len(samples) / OUTPUT_SAMPLE_RATE
                            if samples.size:
                                if args.sentence_buffering:
                                    sentence_audio_chunks.append(samples)
                                else:
                                    queue_speaker_audio(samples)
                                previous_speech_audio = True
                        sentence_boundary = False
                        if args.sentence_buffering and not delegate_touched:
                            sentence_text += text_out
                            sentence_boundary = bool(SENTENCE_END_RE.search(sentence_text))
                        if args.sentence_buffering:
                            if sentence_boundary or end_of_turn or is_listen or delegate_touched:
                                flush_sentence()
                        if not delegate_touched:
                            if text_out:
                                turn_text_parts.append(text_out)
                            turn_audio_seconds += audio_seconds
                        if (end_of_turn or is_listen or delegate_touched) and previous_speech_audio:
                            tail = np.zeros(
                                round(args.playback_tail_seconds * OUTPUT_SAMPLE_RATE),
                                dtype=np.float32,
                            )
                            queue_reply_ending(
                                queue_speaker_audio,
                                reply_end_cue if not delegate_touched else None,
                                tail,
                            )
                            previous_speech_audio = False
                        if (end_of_turn or is_listen) and turn_text_parts:
                            print(
                                f"\n[turn complete] text={''.join(turn_text_parts)!r} "
                                f"generated_audio={turn_audio_seconds:.2f}s",
                                flush=True,
                            )
                            turn_text_parts.clear()
                            turn_audio_seconds = 0.0
                            if turn_retention is not None:
                                eviction = turn_retention.completed_turn(duplex.decoder)
                                if eviction:
                                    print(
                                        f"[context] dropped {eviction['units']} old audio units "
                                        f"({eviction['tokens']} tokens); "
                                        f"cache={eviction['cache']} tokens; "
                                        f"keeping last {args.keep_last_turns} completed turns",
                                        flush=True,
                                    )
                        tool_seconds = 0.0
                        if delegate_body is not None:
                            tool_started = time.monotonic()
                            tool_name = "unknown"
                            try:
                                request = json.loads(delegate_body)
                                tool_name = request.get("tool") if isinstance(request, dict) else None
                                if tool_name == "weather_by_zip" and run_weather_delegate:
                                    tool_result = run_weather_delegate(delegate_body)
                                else:
                                    tool_result = execute_local_tool(
                                        delegate_body, enabled_tools, timers
                                    )
                            except (ValueError, RuntimeError, TypeError) as exc:
                                tool_result = f"Tool request failed: {exc}"
                            pending_backend_texts.append(
                                f"<backend>{tool_result}</backend>"
                            )
                            tool_seconds = time.monotonic() - tool_started
                            print(f"\n[tool:{tool_name or 'unknown'}] {tool_result}", flush=True)
                        if args.control_stdin:
                            print(f"[meter] mic={mic_dbfs:.0f}dBFS", flush=True)
                        if is_listen and not delegate_touched:
                            quiet_chunks += 1
                            if mic_dbfs > -50:
                                print(
                                    f"[input audio] speech reached model; mic={mic_dbfs:.0f}dBFS "
                                    f"chunk={chunk_index}",
                                    flush=True,
                                )
                            if quiet_chunks == 1 or quiet_chunks % 30 == 0:
                                print(
                                    f"[listening] {quiet_chunks}s since last speech; "
                                    f"mic={mic_dbfs:.0f}dBFS chunk={chunk_index}",
                                    flush=True,
                                )
                        else:
                            quiet_chunks = 0
                            print(
                                f"[audio chunk {chunk_index}] state={'delegate' if delegate_touched else 'speak'} "
                                f"mic={mic_dbfs:.0f}dBFS "
                                f"text={text_out!r} audio={audio_seconds:.2f}s "
                                f"tokens={result.get('n_tokens', 0)}/{args.max_new_tokens_per_chunk} "
                                f"tts_tokens={result.get('n_tts_tokens', 0)}/{tts_limit} "
                                f"final_tts_min={result.get('tts_final_min_applied', 0)} "
                                f"turn_end={end_of_turn} "
                                f"inference={inference_seconds:.2f}s tool={tool_seconds:.2f}s "
                                f"llm={result.get('cost_llm', 0.0):.2f}s "
                                f"tts={result.get('cost_tts', 0.0):.2f}s "
                                f"vocoder={result.get('cost_token2wav', 0.0):.2f}s "
                                f"speaker_queue={speaker_chunks.qsize()}",
                                flush=True,
                            )
                        if turn_retention is not None:
                            for eviction in turn_retention.enforce_safety_limit(duplex.decoder):
                                print(
                                    f"[context] {eviction['reason']}: dropped "
                                    f"{eviction['units']} units ({eviction['tokens']} tokens); "
                                    f"cache={eviction['cache']} tokens",
                                    flush=True,
                                )
                    if dropped_mic_blocks:
                        print(
                            f"\n[latency] dropped {dropped_mic_blocks} mic blocks",
                            file=sys.stderr,
                            flush=True,
                        )
                        dropped_mic_blocks = 0
                    if (aec_io is not None and chunk_index >= last_aec_report_chunk + 30):
                        last_aec_report_chunk = chunk_index
                        print(
                            f"[AEC] reference={aec_io.reference_matched}/{aec_io.reference_requested} "
                            f"speaker_blocks={aec_io.reference_active_blocks} "
                            f"echo_blocks_rejected={aec_io.echo_guard.suppressed_blocks} "
                            f"missing_adc_timestamps={aec_io.missing_adc_timestamps} "
                            f"mic_blocks_dropped={aec_io.dropped_mic_blocks}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        if request_stop.is_set():
            print("\nStopping.", flush=True)
        if args.sentence_buffering and sentence_audio_chunks:
            flush_sentence()
        if transcriber is not None:
            transcriber.close()
        stop.set()
        timers.close()
        try:
            while speaker.is_alive():
                speaker.join(timeout=0.2)
        except KeyboardInterrupt:
            print("Playback interrupted; exiting.", flush=True)


if __name__ == "__main__":
    run(parse_args())
