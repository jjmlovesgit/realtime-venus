# Realtime Venus Audio — Windows live voice console

A local, full-duplex voice experiment built on the **Realtime-Venus-Audio 9B** checkpoint. This fork adds a Windows desktop console for live microphone input, streamed speaker output, voice interruption, and playback diagnostics. The model runs on your GPU; the UI, VAD, diagnostic transcription, and audio routing run locally.

This setup was tested on **Windows 11, an NVIDIA GeForce RTX 5090, Python 3.10, and PyTorch 2.11.0 + CUDA 12.8**. It uses the **Audio** checkpoint, not Omni. The [upstream Realtime-Venus project](https://github.com/inclusionAI/Realtime-Venus) provides the model and core inference code.

## Demo reel

**[Watch the 3:19 live console demo (MP4, 37 MB)](demos/venus-demo.mp4)** · [Direct video file](https://raw.githubusercontent.com/jjmlovesgit/realtime-venus/main/demos/venus-demo.mp4)

The recording shows the console and streamed voice interaction. The video is an intentional demo asset; model weights, session recordings, logs, and other generated output are excluded from Git.

## What the console adds

- Live microphone capture and streamed 24 kHz speech playback, with a clickable oscilloscope display to start or stop a session.
- Silero voice activity detection, optional WebRTC echo cancellation, and an unmuted microphone for interruption while Venus speaks.
- Editable full system prompt, device selection, generation and playback controls, and short conversation memory.
- Local diagnostic speech transcription, color coded session log, timing metrics, and optional saved reply audio for investigating clipped speech.
- Optional calculator, countdown timer, and ZIP code weather tools. Weather requires internet access.
- A quiet radio static loop with a Squelch control and separate microphone/reply end cues.

## Windows 11 / RTX 5090 setup

Run these commands in **PowerShell**. The examples assume the repository is at `C:\Projects\Realtime-Venus-Audio`; use your own clone path if different. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or another Conda distribution and [FFmpeg](https://ffmpeg.org/download.html), then ensure `ffmpeg -version` works in the active terminal. Use headphones for the best interruption and echo cancellation results.

### 1. Clone and create a Python 3.10 environment

```powershell
git clone https://github.com/jjmlovesgit/realtime-venus.git C:\Projects\Realtime-Venus-Audio
Set-Location C:\Projects\Realtime-Venus-Audio
conda create -n venus-audio python=3.10 -y
conda activate venus-audio
python -m pip install --upgrade pip
```

If you already have this project and the `venus-audio` environment, activate it and skip the clone and environment creation.

### 2. Install the CUDA 12.8 PyTorch build

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

The tested output begins with `2.11.0+cu128` and names the RTX 5090. The wheel command follows [PyTorch's official version-specific instructions](https://pytorch.org/get-started/previous-versions/).

### 3. Install the Audio and console dependencies

```powershell
python -m pip install -c constraints-windows-cu128.txt "transformers==4.51.3" "accelerate>=1.3,<2" "numpy>=1.24,<2" "sentencepiece==0.2.0" "safetensors>=0.5,<1" "resampy>=0.4,<0.5" "setuptools>=68,<81" "minicpmo-utils[tts]==1.0.6" "librosa==0.9.0" sounddevice soundfile scipy huggingface_hub PyYAML websockets
python -m pip install -c constraints-windows-cu128.txt -r requirements-ui.txt -r requirements-vad.txt -r requirements-stt.txt -r requirements-aec.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
```

The constraint file pins the three CUDA wheels and the checkpoint's compatible Transformers version. **Do not install the root `requirements.txt` or `demos/requirements.txt` into this environment**: the upstream demo dependencies pin `torch==2.4.0` and can replace the CUDA 12.8 build. If dependency resolution reports a conflict, keep the installed CUDA wheels and resolve that conflict rather than removing the constraints.

### 4. Download the Audio checkpoint and diagnostic STT model

```powershell
python download_models.py --model audio --local-dir .
python download_stt_model.py
```

The checkpoint lands in `Realtime-Venus-Audio\`; the small Vosk model lands in `models\`. Both are ignored by Git. The Hugging Face download resumes from its cache after a network interruption. You can skip the Vosk download and disable **Transcribe mic in log** if you do not need a local input transcript.

### 5. Run the console

```powershell
python live_mic_duplex.py --list-devices
python live_mic_ui.py
```

Choose the microphone and speaker in **Controls → Audio routing**, then click **Start listening** or the oscilloscope display. Device changes and most generation/playback changes apply when you stop and start a new session. A voice reference can be selected in **Playback → Voice WAV**; when left blank, the project uses `assets/voice_reference.wav` if you place one there, otherwise the checkpoint reference. Your private voice file stays out of Git.

For a terminal-only run, use `python live_mic_duplex.py`. See the [full Windows live microphone guide](LIVE_MIC_WINDOWS.md) for every switch, playback tuning, and troubleshooting. The [test plan](TEST_PLAN.md) covers functional checks.

## Project files

| Path | Purpose |
| --- | --- |
| `live_mic_ui.py` | Windows desktop console and saved controls |
| `live_mic_duplex.py` | Live Audio checkpoint inference and streaming playback |
| `duplex_aec.py`, `mic_vad.py`, `input_stt.py` | Echo cancellation, speech detection, diagnostic transcription |
| `assets/console_background.png`, `assets/mic-off.mp3`, `assets/radio-static-idle.mp3` | Console art and intentional sound assets |
| `demos/venus-demo.mp4` | Shareable demo reel |
| `frontend/Realtime-Venus-Audio/` | Upstream Audio inference examples |

## Credit and license

Realtime-Venus and its Audio checkpoint are from the [Venus team at Ant Group and Tsinghua University](https://github.com/inclusionAI/Realtime-Venus). This fork retains the upstream source and its [Apache 2.0 license](LICENSE). Download the model under its own published terms from [inclusionAI/Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus).
