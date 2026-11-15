# Realtime-Venus-Audio local test plan

## Goals we have discussed

- Run the **audio-only** Realtime-Venus model locally on Windows 11 and the RTX 5090.
- Have a real back-and-forth voice conversation through the microphone and speakers, with the model listening while it speaks.
- Keep the model and audio processing local. The model download is complete at `Realtime-Venus-Audio/`.
- Measure whether the interaction feels real time before adding features.
- Later, investigate tool calling, VAD, and a wake word. These are follow-on goals; first establish a reliable voice baseline.
- A Windows SAPI voice was discussed as an option, but the model has its own speech decoder. Replacing its audio with SAPI may affect latency, timing, and interruption behavior, so defer that until after baseline testing.

## Phase 1: Protect and verify the environment

- [x] Create and activate the Python 3.10 `venus-audio` environment.
- [x] Confirm PyTorch reports `2.11.0+cu128` and `NVIDIA GeForce RTX 5090`.
- [x] Download only the Audio checkpoint.
- [x] Confirm `Realtime-Venus-Audio/config.json` and all four model weight shards exist (about 18.8 GB on disk).
- [x] Install the audio inference dependencies without replacing the CUDA 12.8 PyTorch build.
- [x] Recheck `torch.__version__`, `torch.cuda.is_available()`, and the GPU name after dependency installation.
- [x] Confirm FFmpeg is available on PATH and `sounddevice` can list audio devices.

**Dependency caution:** the repository's full `requirements.txt` includes a PyTorch 2.4.0 pin. Do not install it unchanged into this environment. After installing dependencies, verify that pip did not replace the working `2.11.0+cu128` build.

## Phase 2: Establish a model baseline

- [x] Run the upstream offline audio example with its bundled sample; it answered the sample successfully.
- [x] Run the upstream recorded full-duplex example and confirm it writes a playable WAV (24 kHz, 29.04 seconds, nonzero audio).
- [ ] Record model load time, peak VRAM, and whether the 5090 stays responsive.
- [ ] Save the exact commands, package versions, and any errors before proceeding.

## Phase 3: Test live conversation

- [ ] List Windows audio devices and select the intended microphone and headphone/speaker output.
- [ ] Add structured timing reports to the live script before the first session; use a monotonic clock and log timestamps to a CSV or JSONL file.
- [ ] Run `live_mic_duplex.py` for a short, simple exchange.
- [ ] Confirm the model hears speech, prints text, and plays generated speech.
- [ ] Test several turns, including a pause and a correction while the model is speaking.
- [ ] Use headphones initially to avoid feedback from speaker audio into the microphone.
- [ ] Report time from first captured speech chunk to first response text (TTFT).
- [ ] Report time from first captured speech chunk to first audible response audio (TTFA), separately from TTFT.
- [ ] Report per-chunk prefill and generation duration, chunk audio duration, and real-time factor (processing seconds divided by audio seconds).
- [ ] Report per-turn end-of-user-speech to first response text/audio latency, total response duration, dropped mic/speaker chunks, and GPU peak VRAM.
- [ ] Measure interruption response: time from speaking over playback to the model yielding or changing its response.
- [ ] Note missed speech, clipped output, audio drop counters, and GPU/VRAM use.
- [ ] Repeat with a longer conversation and decide whether processing keeps pace with live audio.

**Baseline success criteria:** at least five consecutive spoken turns work; generated speech is audible and understandable; the model accepts a follow-up without restarting; no unbounded audio backlog develops; and TTFT, TTFA, chunk real-time factor, per-turn latency, peak VRAM, and interruption response are reported with enough timestamps to reproduce the measurements.

## Phase 4: Improve interaction only after baseline

- [ ] Tune chunking and playback if latency or audio quality needs improvement.
- [ ] Test the opt-in local calculator and countdown timer with harmless spoken requests; confirm the model emits each delegate call, speaks tool results, cancels by timer ID, and delivers an expiration alert while the script is still running.
- [ ] Add VAD to handle silence and turn boundaries, then measure false starts and missed speech.
- [ ] Add a local wake word only if always-listening activation is desired; measure false activations.
- [ ] Test weather tool calling separately and verify both the delegate request and returned result.
- [ ] Reassess native speech versus SAPI only if a different Windows voice is still wanted.
- [ ] Compare the bundled voice with one clean 5–15 second custom reference WAV; note accent, similarity, intelligibility, and added TTS initialization time.

## Current status

The checkpoint download succeeded. The environment has Python 3.10, PyTorch `2.11.0+cu128`, Transformers 4.51.3, and the RTX 5090. Offline and recorded duplex inference both work. The duplex scripts initialize TTS with a reference WAV; a small compatibility shim reads WAV files through SoundFile because TorchCodec's Windows FFmpeg DLL loader fails on this machine. `assets/voice_reference.wav` is the selected project default (6.26 seconds, mono, 24 kHz). The live microphone test and real-time performance remain unverified. Timing instrumentation is a prerequisite for that live test.
