# Venus ↔ avatar handoff

Updated 2026-11-15. This file records the exact local setup so the next session can resume without reconstructing it from chat.

## Which folders and environments matter

| Role | Working folder | Conda environment | Verified build |
| --- | --- | --- | --- |
| Live Venus Audio app | `C:\Projects\Realtime-Venus-Audio` | `venus-audio` | Python 3.10, PyTorch `2.11.0+cu128`, RTX 5090 |
| Avatar app we actually launch | `G:\FlashIMTalker` | `flashrtRTC` | Python 3.11, PyTorch `2.7.0+cu128`, RTX 5090 |
| Earlier avatar development copy | `C:\Projects\FlashIMTalkerV2` | Tested with `flashrt311v2` and `imtalker-rtx5090` | **Do not confuse this with the running `G:` copy.** |

The `G:` avatar repository has substantial pre-existing local changes and downloaded checkpoints. The new bridge files there are local; no avatar repository commit or push was made. The Venus repository is `https://github.com/jjmlovesgit/realtime-venus`.

## Start each app

**Avatar video endpoint** (PowerShell window 1):

```powershell
Set-Location G:\FlashIMTalker
conda activate flashrtRTC
python python_venus_avatar_server.py
```

This focused launcher serves `http://127.0.0.1:8002` and starts only the avatar model, its image, and GPU render queue. It skips the avatar app's own Whisper, wake word, PocketTTS warmup, LM Studio/OpenClaw checks, and GPU keep-alive. If the original full server (`python python_avatar_server_electron.py`) is already using port 8002, stop it with Ctrl+C first. Changing source files requires a restart.

**Venus voice UI** (PowerShell window 2):

```powershell
Set-Location C:\Projects\Realtime-Venus-Audio
conda activate venus-audio
python live_mic_ui.py
```

The focused avatar launcher is in `G:\FlashIMTalker\python_venus_avatar_server.py`. The new API routes are in that folder's `python_avatar_server_electron.py`; their detailed contract and a manual curl test are in `G:\FlashIMTalker\VENUS_AUDIO_ENDPOINT.md`. The avatar route tests are in `G:\FlashIMTalker\tests\test_venus_audio_endpoint.py`.

## What works now

- `POST /avatar/audio` accepts one mono PCM16 WAV chunk (8–48 kHz, 0.1–15 s, at most 2 MB) from localhost. Venus's 24 kHz WAV needs no caller-side resampling. It queues the already-generated speech for avatar rendering and returns a job ID. It does **not** call the avatar app's Whisper, LLM, or PocketTTS.
- `GET /avatar/audio/{job_id}` returns `queued`, `generating`, `ready`, or `error`. A ready job has `video_url` under `/chunks/`, `audio_seconds`, and `generation_seconds`. `DELETE` removes the completed job and MP4. The input WAV is removed automatically.
- The MP4 currently carries the uploaded speech as AAC. Keep Venus responsible for speaker playback and **mute the avatar video** when it is eventually shown in the Venus UI.
- Four isolated endpoint/startup tests passed in `flashrtRTC` with a fake renderer. A real short saved Venus reply also produced a valid 512×512, 25 fps H.264/AAC MP4 on the original full avatar server.

## Timing measured so far

| Input speech | Fast avatar render | Notes |
| --- | --- | --- |
| 1.5 s | 0.97 s | Valid 1.5 s MP4, visually inspected |
| 3.0 s | 1.72 s | Valid 3.0 s MP4 |

These measurements used the **full** avatar server in `G:\FlashIMTalker`, with Venus **not loaded concurrently**. Both test jobs and their MP4s were deleted. The focused startup has passed import and mocked lifecycle tests but has not yet been measured with the real model. These figures do not determine the final lip-sync buffer: collecting a speech chunk and sharing the 5090 with Venus may add latency.

## Next integration steps

1. Start the focused avatar server and Venus together. Check that `GET http://127.0.0.1:8002/openapi.json` lists `/avatar/audio`, then render saved 1–3 s Venus speech chunks while both models are resident. Record GPU memory, `generation_seconds`, and any slowdown of Venus audio.
2. If the shared-GPU test fits and stays responsive, copy **clean model speech only** from Venus's `queue_speaker_audio` path to a nonblocking avatar upload queue. Exclude the radio static, cue sounds, and tail silence. Keep microphone capture and speaker playback inside Venus.
3. Add a video-capable display to the Venus UI. Play avatar MP4s muted, discard old queued video on interruption, and measure the gap from speech generation to visible lip motion. Decide the playback buffer from those measurements; do not assume a fixed value from the isolated tests.

The bridge is **not yet wired into `live_mic_duplex.py` or `live_mic_ui.py`**. The avatar endpoint can currently be tested manually with saved reply WAV files. No user voice reference, model weights, logs, or generated media should be committed to the Venus repository.
