# Live microphone on Windows

`live_mic_duplex.py` feeds 16 kHz mono microphone audio to the Audio checkpoint
using the same `streaming_prefill` and `streaming_generate` calls as the upstream
file example. It plays generated 24 kHz speech through the selected speaker.
The CLI normally builds a prompt with English and answer instructions around
`--system-prompt`. Pass `--exact-system-prompt --system-prompt "..."` to send
only the supplied text.

Run from this repository's root in a Python 3.10 environment with the Audio
checkpoint at `Realtime-Venus-Audio/`:

```powershell
python -m pip install sounddevice
python -m pip install --no-deps -r requirements-stt.txt
python -m pip install --no-deps -r requirements-vad.txt
python download_stt_model.py
python live_mic_duplex.py --list-devices
python live_mic_duplex.py
```

The small Vosk English model transcribes the microphone locally for diagnostics.
The `[input STT]` lines are an independent estimate of what was spoken; Venus
does not provide its internal input transcript. Use `--no-input-stt` to disable
this extra transcription. The UI has a **Transcribe mic in log** switch.
**Mute mic while Venus speaks** is off by default. When enabled, it feeds silence to Venus and
the diagnostic transcriber from the start of speaker playback until the queued
reply finishes, preventing headset echo from becoming a new turn. Voice
interruptions are unavailable while it is on. Turn it off in the UI or pass
`--no-mute-mic-during-playback` to test overlapping conversation.
The UI also enables **Silero VAD** by default. It runs locally on the CPU and
replaces detected non-speech, often including breathing, with silence before feeding
the model and input transcript. Its threshold defaults to 0.6; raise it to
reject more noise or lower it if quiet speech is missed. It keeps about 160 ms
before detected speech and 320 ms afterward to protect word boundaries. The
CLI enables it with `--silero-vad`. The package is installed with `--no-deps`
so pip does not replace the CUDA PyTorch build.

For a control window with device selection, playback and generation settings,
optional tools, Start/Stop, and a live log, run:

```powershell
python -m pip install --no-deps -r requirements-ui.txt
python live_mic_ui.py
```

The UI uses rounded controls. Click a section title to collapse or expand it;
that layout is restored on the next launch. **Clear log** empties the visible
Session activity log without stopping the model.
The voice display at the top is also a Start/Stop control. The supplied console
artwork is the darkened backdrop for that display and the surrounding controls.
The outer **Controls** panel can collapse while the voice display and Session
activity log remain visible.
The log marks microphone transcripts in blue (**YOU**) and Venus reply text in
green (**VENUS**). Timing and audio diagnostics remain muted. Saved log files
retain plain text.
Session activity stays in a lower pane while the controls scroll above it.
Drag the visible bar labeled **DRAG HERE TO RESIZE LOG** up or down to resize
the pane. The size is restored when the UI opens again.
Hover over a Playback or Generation label or control for an explanation of its setting.
Each UI session is saved under `logs/`. Mic levels continue updating the meter,
while repetitive listen chunks stay out of the log; long listening periods get
a short status line every 30 seconds.

The System prompt inside Controls is the **entire** text sent to
Venus. You can edit the prefix, English rule, answer rule, story rule, and any
other wording. Click **Save prompt** to persist it in
`live_mic_ui_settings.json`; changes take effect when the next session starts.
Older saved prompts are expanded once to preserve their previous effective
text. The session log prints the exact prompt sent to the model. Tool checkboxes
enable tool execution, but with a fully editable prompt they do not silently
add guidance; use **Add enabled tool rules** to insert editable instructions.

**Conversation memory → Keep last turns** defaults to 3 in the UI. One turn is
a completed user-and-Venus exchange, not a one-second audio chunk. After the
fourth completed reply, the app removes the oldest exchange from the language
model cache and keeps the system prompt. Set 0 to disable this trimming. The
CLI option is `--keep-last-turns 3`. If a few unusually long turns push the
cache past 36,000 tokens, a safety trim brings it below 30,000, even if that
means retaining fewer than three turns. Removed turns are not summarized; the
session log reports each eviction as `[context]`.

Playback has separate **Text tokens** and **TTS tokens** controls. The text
limit does not increase the amount of speech generated; the TTS limit does.
The session log prints the active values and speech chunks' text/TTS token counts.
For a clipped ending, enable **Save reply audio for diagnosis** (on by
default in the UI). Realtime playback continues normally; each completed reply
gets its own `logs/reply-*.wav` file. Compare that file with the speaker
output. A `[playback complete]` line confirms the queued reply reached the
speaker DAC. The CLI option is `--save-last-reply-debug`.
**Final TTS min** sets a minimum number of speech tokens for the final,
text-empty audio flush. The default is 32; set it to 0 to disable this
experimental last-word setting. Its effect appears as `final_tts_min` in the
speech-chunk log. It cannot restore words missing from the generated text.
**Final TTS max** separately limits that last audio flush. Its default is 128,
so a final phrase has more room than ordinary chunks. If the final log shows
`tts_tokens` one below this limit and speech is missing, the flush may still
be capped; raise Final TTS max and retry.
Changing playback controls during a session requires Stop, then Start. Tail
silence keeps the speaker stream open after a turn but cannot synthesize words
that are missing from the model's audio output.
The app plays `assets/radio-static-idle.mp3` quietly when no session is active
and while a live session has no queued speech or cue. The loop trims quiet edges
in memory and crossfades its join; the source MP3 stays intact. Playback's
**Squelch** slider sets its target level from -65 to -25 dBFS, starting at -48.
The app-side idle loop responds after a short pause when the slider moves; the
live session uses the new value after Stop and Start. Disable **Idle radio
static** to turn it off. The static is excluded from saved reply WAV files and
from the voice display's speech meter. AEC uses the actual speaker output as
its reference, including the static bed.

Both the microphone speech-end cue and the reply-end cue use
`assets/mic-off.mp3`. They are separate speaker sounds rather than model speech,
and are excluded from saved reply WAV files. The microphone cue uses Silero's
speech endpoint; when VAD filtering is off, Silero detects that endpoint
without changing the audio sent to Venus. The cue asset is loaded once per
session. **Mic gain** defaults to -2 dB and changes microphone loudness, not
pitch; adjust it in Audio routing or with `--mic-gain-db`.
Stop finishes queued playback before returning to Start; closing the window also
stops the session. A second close forces the process to exit.

Pass `--input-device INDEX --output-device INDEX` to select devices. Use the
indices printed by `--list-devices`; the default Windows devices on this PC
support 16 kHz input and 24 kHz output. A particular WASAPI device may reject
one of those sample rates. Use headphones to avoid speaker audio entering the
microphone. Press Ctrl+C to stop.

The project uses `assets/voice_reference.wav` by default when present; otherwise it
uses the checkpoint's built-in reference. To select another voice, pass
`--voice-reference C:\path\to\voice.wav`.
Use a clean recording of one speaker, ideally about 5–15 seconds, with no music,
other voices, or long silence. The checkpoint's `assets/HT_ref_audio.wav` is used
when neither a project reference nor this option is present.

By default, generated audio plays as soon as each chunk is ready. To wait for
sentence-ending punctuation, end-of-turn, or a return to listening, use:

```powershell
python live_mic_duplex.py --sentence-buffering
```

Pass `--no-sentence-buffering` to explicitly select immediate chunk playback;
omitting both flags does the same.

Sentence buffering joins the audio before playback and removes exact leading
zero padding from short TTS chunks. It delays the first sound until the sentence
boundary. In either mode, playback preserves queued audio if the speaker falls
behind. A full speaker queue can delay inference and cause the microphone queue
to drop old input; `speaker_queue` and `[latency]` show when this happens. One
Ctrl+C stops generation and drains queued playback; a second interrupts playback.
The script adds 0.5 seconds of silence immediately after the model ends a spoken
turn to let the last audio chunk finish. A `[turn complete]` line shows the full
generated text and audio duration; if it omits a word, the model did not generate
that word.
The default speech generation allowance is 40 tokens per chunk, which completed
a spoken countdown from ten in the local trial. Use
`--max-new-tokens-per-chunk NUMBER` to adjust it; larger values can increase
latency.

Enable the local calculator and countdown timer with:

```powershell
python live_mic_duplex.py --calculator-tool --timer-tool
```

Ask for arithmetic, say “set a timer for two minutes,” or cancel a timer by its
spoken ID. Timer calls require the model to provide a short phrase explicitly
requesting the start or cancellation; the local runtime rejects calls without
that wording. The calculator accepts numeric expressions with parentheses and
`+`, `-`, `*`, `/`, `//`, `%`, and `**`. Timer durations are limited to 24 hours.
The script must remain running for timer alerts; when a timer expires, its notice
is returned to Venus for spoken delivery. These tools run locally. The model must
emit its native `<delegate>` tool request for a tool to run.

The optional ZIP-code weather tool is enabled separately with `--weather-tool`.
Ask for current weather in a five-digit U.S. ZIP code (for example, “What's the
weather in 10001?”). It uses Open-Meteo without an API key and needs internet
access. The free API is for non-commercial use; Open-Meteo attribution is required
by its data licence.

To download the Audio checkpoint, follow
[the upstream Audio guide](frontend/Realtime-Venus-Audio/README.md). The current
`demos/requirements.txt` pins `torch==2.4.0`, which conflicts with the CUDA 12.8
PyTorch setup intended for an RTX 5090. Installing `requirements.txt` after a
newer CUDA build can downgrade PyTorch. Check `python -c "import torch;
print(torch.__version__, torch.cuda.get_device_name(0))"` before running.

The script is designed for live use: it discards old microphone buffers if
processing falls behind and reports those drops. Dropped input may affect the
model's conversation context.

For voice interruption while Venus speaks, enable **Cancel Venus speaker echo**
in the UI and leave **Mute mic while Venus speaks** off. Echo cancellation uses
WebRTC AEC3 with the exact speaker signal on one 48 kHz duplex stream; Silero
VAD remains a separate speech detector. Both selected devices must support
48 kHz. Install the optional processor in the active `venus-audio` environment:

```powershell
python -m pip install --no-deps -r requirements-aec.txt
```

The same option is available on the CLI as `--echo-cancellation`. This does
not alter the CUDA 12.8 PyTorch installation. If the stream cannot open,
choose compatible input and output devices or turn the switch off for comparison.
The session log reports `[AEC] reference=...` and `speaker_blocks=...` every 30
audio chunks. On this
Windows headset, PortAudio does not supply a microphone capture timestamp;
the app uses the stream clock and measured input latency instead. The reference
count should rise during a session, while `speaker_blocks` rises when Venus
plays. With no speaker audio, the mic bypasses AEC so its speech remains intact.
Low-level residual echo is suppressed
while speaker audio plays, but clear microphone speech can still interrupt.
If speech reaches the model but it keeps choosing `listen`, the UI's **Favor
reply after speech** switch briefly lowers the listen token's score after
VAD-detected speech ends. It makes at most two attempts and does not inject
words into the conversation. The log shows `[response nudge]` when used.
