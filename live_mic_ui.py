#!/usr/bin/env python3
"""Local Windows controls for the Realtime-Venus live microphone script."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import sounddevice as sd
from PIL import Image, ImageOps, ImageTk

from live_mic_duplex import (
    CALCULATOR_INSTRUCTIONS, TIMER_INSTRUCTIONS, WEATHER_INSTRUCTIONS,
    build_legacy_system_prompt,
)
from idle_audio import UIIdlePlayer
from voice_scope import VoiceScope


PROJECT = Path(__file__).resolve().parent
SCRIPT = PROJECT / "live_mic_duplex.py"
SETTINGS = PROJECT / "live_mic_ui_settings.json"
DEFAULT_SETTINGS = PROJECT / "live_mic_ui_defaults.json"
BACKGROUND_ART = PROJECT / "assets" / "console_background.png"
BG = "#0B1018"
PANEL = "#151E2B"
PANEL_RAISED = "#1D2A3A"
TEXT = "#F0F5FA"
MUTED = "#9AAABD"
ACCENT = "#7DE2C6"
GOLD = "#EFC786"

GENERATION_HELP = {
    "decode_mode": (
        "Sampling can vary the wording of a reply. Greedy chooses the highest-scoring "
        "next token. Temperature, Top K, and Top P affect sampling only."
    ),
    "temperature": (
        "Sampling randomness. Lower values usually make replies steadier; higher "
        "values allow more variation. Ignored in greedy mode."
    ),
    "top_k": (
        "In sampling mode, consider only the K highest-scoring next tokens. "
        "Smaller values narrow the choices. Ignored in greedy mode."
    ),
    "top_p": (
        "In sampling mode, keep the smallest group of next tokens whose total "
        "probability reaches this value. Smaller values narrow the choices. "
        "Ignored in greedy mode."
    ),
    "listen_prob_scale": (
        "Adjusts the model's listen-token score, which can change when it starts "
        "or stops speaking. 1.0 leaves that score unchanged. The effect of higher "
        "or lower values can vary with context."
    ),
}

PLAYBACK_HELP = {
    "sentence_buffering": (
        "Hold generated audio until sentence punctuation or the model ends its turn, "
        "then play that sentence together. This can sound smoother but delays playback."
    ),
    "max_new_tokens_per_chunk": (
        "Maximum text tokens the model may produce for each one-second microphone chunk. "
        "Higher values may allow longer phrases but can add latency. This is not a "
        "total story-length limit."
    ),
    "playback_tail_seconds": (
        "Seconds of silence appended after a spoken turn so the speaker can finish "
        "its last buffer. This cannot restore a word the model did not synthesize."
    ),
    "idle_static": (
        "Play the supplied radio static while the app is idle, and loop it quietly "
        "during a live session whenever no reply or cue is playing. "
        "It fades out for speech and is excluded from saved reply WAV files."
    ),
    "idle_static_level_dbfs": (
        "Target loudness of the idle radio static in dBFS. More negative is quieter. "
        "Start at -48; use -55 for a subtler background."
    ),
    "mic_gain_db": (
        "Level of the microphone sent to Venus. -2 dB makes your input slightly "
        "quieter; it does not change pitch. Changes apply to the next session."
    ),
    "speaker_queue_size": (
        "Maximum number of generated audio chunks waiting for the speaker. A larger "
        "queue tolerates playback lag but can make responses arrive later."
    ),
    "tts_max_new_tokens_per_chunk": (
        "Maximum speech-audio tokens synthesized per model chunk. More tokens can "
        "allow longer audio for a phrase, but may increase generation time. It does "
        "not make the model write a longer answer."
    ),
    "tts_final_min_tokens": (
        "Minimum speech-audio tokens on a final, text-empty flush. A modest value "
        "can help finish a clipped last syllable; larger values may add trailing "
        "sound. Use 0 to disable."
    ),
    "tts_final_max_tokens": (
        "Maximum speech-audio tokens for the final flush of a reply. Raise this "
        "if the final chunk reaches the limit and words are cut off. A higher "
        "value can delay the end of a reply."
    ),
    "voice_reference": (
        "WAV clip used as the reference for Venus's spoken voice. Leave blank to use "
        "the project's default voice clip. This does not change microphone input."
    ),
}


class HoverTip:
    """A delayed tooltip that also works over CustomTkinter's child widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.pending_show = None
        self.pending_hide = None
        self.popup = None
        self._bind_children(widget)

    def _bind_children(self, widget):
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        for child in widget.winfo_children():
            self._bind_children(child)

    def _cancel(self, attr):
        after_id = getattr(self, attr)
        if after_id is not None:
            self.widget.after_cancel(after_id)
            setattr(self, attr, None)

    def _enter(self, _event):
        self._cancel("pending_hide")
        if self.popup is None and self.pending_show is None:
            self.pending_show = self.widget.after(500, self.show)

    def _leave(self, _event):
        self._cancel("pending_show")
        self._cancel("pending_hide")
        self.pending_hide = self.widget.after(100, self._hide_if_outside)

    def _hide_if_outside(self):
        self.pending_hide = None
        x, y = self.widget.winfo_pointerxy()
        left, top = self.widget.winfo_rootx(), self.widget.winfo_rooty()
        if left <= x < left + self.widget.winfo_width() and top <= y < top + self.widget.winfo_height():
            return
        self.hide()

    def show(self):
        self.pending_show = None
        if self.popup is not None or not self.widget.winfo_exists():
            return
        popup = tk.Toplevel(self.widget)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(bg=BG)
        popup.attributes("-topmost", True)
        frame = ctk.CTkFrame(popup, fg_color=PANEL_RAISED, bg_color=BG,
                             corner_radius=12, border_width=1, border_color="#426077")
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text=self.text, text_color=TEXT, fg_color="transparent",
                     font=("Segoe UI", 10), justify="left", anchor="w",
                     wraplength=290).pack(padx=12, pady=9)
        popup.update_idletasks()
        pointer_x, pointer_y = self.widget.winfo_pointerxy()
        x = min(pointer_x + 14, popup.winfo_screenwidth() - popup.winfo_reqwidth() - 8)
        y = min(pointer_y + 18, popup.winfo_screenheight() - popup.winfo_reqheight() - 8)
        popup.geometry(f"+{max(0, x)}+{max(0, y)}")
        popup.deiconify()
        self.popup = popup

    def hide(self, _event=None):
        self._cancel("pending_show")
        self._cancel("pending_hide")
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None


class LiveMicUI:
    def __init__(self, root):
        self.root = root
        self.root.title(" ")
        self.root.geometry("1120x850")
        self.root.minsize(950, 690)
        self.proc = None
        self.closing = False
        self.log_items = queue.Queue()
        self.idle_player = UIIdlePlayer(
            PROJECT / "assets" / "radio-static-idle.mp3",
            on_error=lambda exc: self.log_items.put(f"[idle audio] {exc}\n"),
        )
        self.log_panel_height = 260
        self._log_drag_origin = None
        self.sections = {}
        self.tooltips = []
        try:
            self.wallpaper_source = Image.open(BACKGROUND_ART).convert("RGB")
        except OSError:
            self.wallpaper_source = None
        self.wallpaper_photo = None
        self.wallpaper_size = None
        self._idle_sync_job = None
        self.vars = {
            "input_device": tk.StringVar(value="Default"),
            "output_device": tk.StringVar(value="Default"),
            "voice_reference": tk.StringVar(value=""),
            "input_stt": tk.BooleanVar(value=True),
            "mute_mic_during_playback": tk.BooleanVar(value=False),
            "echo_cancellation": tk.BooleanVar(value=True),
            "silero_vad": tk.BooleanVar(value=True),
            "vad_threshold": tk.StringVar(value="0.6"),
            "mic_gain_db": tk.StringVar(value="-2"),
            "save_last_reply_debug": tk.BooleanVar(value=True),
            "sentence_buffering": tk.BooleanVar(value=False),
            "max_new_tokens_per_chunk": tk.StringVar(value="40"),
            "tts_max_new_tokens_per_chunk": tk.StringVar(value="64"),
            "tts_final_min_tokens": tk.StringVar(value="32"),
            "tts_final_max_tokens": tk.StringVar(value="128"),
            "playback_tail_seconds": tk.StringVar(value="0.5"),
            "idle_static": tk.BooleanVar(value=True),
            "idle_static_level_dbfs": tk.StringVar(value="-48"),
            "speaker_queue_size": tk.StringVar(value="8"),
            "decode_mode": tk.StringVar(value="sampling"),
            "temperature": tk.StringVar(value="0.7"),
            "top_k": tk.StringVar(value="20"),
            "top_p": tk.StringVar(value="0.8"),
            "listen_prob_scale": tk.StringVar(value="1.0"),
            "reply_after_speech": tk.BooleanVar(value=True),
            "keep_last_turns": tk.StringVar(value="3"),
            "calculator_tool": tk.BooleanVar(value=False),
            "timer_tool": tk.BooleanVar(value=False),
            "weather_tool": tk.BooleanVar(value=False),
        }
        self.status = tk.StringVar(value="Stopped")
        self.level = tk.DoubleVar(value=0)
        self.level_text = tk.StringVar(value="MIC  ·  — dBFS")
        self._style()
        self._build()
        self._load_settings()
        self._sync_squelch_readout()
        self._refresh_devices()
        for key in ("sentence_buffering", "mute_mic_during_playback", "echo_cancellation",
                    "reply_after_speech", "keep_last_turns",
                    "silero_vad", "vad_threshold", "save_last_reply_debug",
                    "max_new_tokens_per_chunk", "tts_max_new_tokens_per_chunk",
                    "tts_final_min_tokens", "tts_final_max_tokens",
                    "playback_tail_seconds", "speaker_queue_size",
                    "idle_static", "idle_static_level_dbfs", "mic_gain_db"):
            self.vars[key].trace_add("write", self._playback_setting_changed)
        for key in ("idle_static", "idle_static_level_dbfs", "output_device"):
            self.vars[key].trace_add("write", self._schedule_idle_sync)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)
        self.root.after(200, self._sync_idle_player)

    def _style(self):
        ctk.set_appearance_mode("dark")
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Card.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))

    def _button(self, parent, text, command, primary=False, width=140, state="normal"):
        return ctk.CTkButton(
            parent, text=text, command=command, width=width, height=36,
            corner_radius=11, fg_color=ACCENT if primary else PANEL_RAISED,
            hover_color="#A5F1DA" if primary else "#304158",
            text_color=BG if primary else TEXT, state=state,
            font=("Segoe UI Semibold" if primary else "Segoe UI", 11),
        )

    def _card(self, parent, title, subtitle):
        card = ctk.CTkFrame(parent, fg_color=PANEL, bg_color=BG, corner_radius=18,
                            border_width=1, border_color="#26374A")
        header = self._button(card, f"{title}   ▾", lambda name=title: self._toggle_section(name), width=100)
        header.configure(fg_color="transparent", hover_color=PANEL_RAISED,
                         anchor="w", height=43, font=("Segoe UI Semibold", 15))
        header.pack(fill="x", padx=9, pady=(7, 0))
        subtitle_label = tk.Label(card, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        subtitle_label.pack(anchor="w", padx=18, pady=(0, 12))
        body = ctk.CTkFrame(card, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 15))
        self.sections[title] = {"header": header, "subtitle": subtitle_label,
                                "body": body, "collapsed": False,
                                "body_pack": {"fill": "both", "expand": True,
                                              "padx": 18, "pady": (0, 15)}}
        return card, body

    def _toggle_section(self, title):
        self._set_section_collapsed(title, not self.sections[title]["collapsed"])
        try:
            self._save_settings()
        except OSError:
            pass

    def _set_section_collapsed(self, title, collapsed):
        section = self.sections.get(title)
        if section is None or section["collapsed"] == collapsed:
            return
        section["collapsed"] = collapsed
        section["header"].configure(text=f"{title}   {'▸' if collapsed else '▾'}")
        if collapsed:
            if section["subtitle"] is not None:
                section["subtitle"].pack_forget()
            section["body"].pack_forget()
        else:
            if section["subtitle"] is not None:
                section["subtitle"].pack(anchor="w", padx=18, pady=(0, 12))
            section["body"].pack(**section["body_pack"])
        if title == "Controls":
            self.root.after(80, self._restore_log_pane_height)

    def _build(self):
        viewport = tk.Frame(self.root, bg=BG)
        viewport.pack(fill="both", expand=True)
        self.log_pane = tk.PanedWindow(
            viewport, orient=tk.VERTICAL, bg="#385069", borderwidth=0,
            sashwidth=6, showhandle=False, opaqueresize=True,
        )
        self.log_pane.pack(fill="both", expand=True)
        controls = tk.Frame(self.log_pane, bg=BG)
        self.log_panel = tk.Frame(self.log_pane, bg=BG)
        self.log_pane.add(controls, minsize=250, stretch="always")
        self.log_pane.add(self.log_panel, minsize=180, stretch="never")
        self.log_pane.bind("<ButtonRelease-1>", self._log_resize_end)
        self.root.after(150, self._restore_log_pane_height)

        canvas = tk.Canvas(controls, bg=BG, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ctk.CTkScrollbar(controls, orientation="vertical", command=canvas.yview,
                                      fg_color=BG, button_color="#304158",
                                      button_hover_color="#46627D", corner_radius=8)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
        outer = tk.Frame(canvas, bg=BG, padx=20, pady=20)
        self.controls_outer = outer
        if self.wallpaper_source is not None:
            self.wallpaper_label = tk.Label(
                outer, bg=BG, borderwidth=0, highlightthickness=0,
            )
            self.wallpaper_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.wallpaper_label.lower()
            outer.bind("<Configure>", self._render_wallpaper)
        window = canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")), add="+")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        self.root.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))

        scope_panel = ctk.CTkFrame(
            outer, fg_color="#0C1B29", bg_color=BG, border_width=1,
            border_color="#3C6470", corner_radius=20,
        )
        scope_panel.pack(fill="x", pady=(0, 17))
        self.scope = VoiceScope(
            scope_panel, self._toggle_scope_session, backdrop_path=BACKGROUND_ART,
        )
        self.scope.pack(fill="x", padx=10, pady=10)
        self.scope_status = ctk.CTkLabel(
            scope_panel, textvariable=self.status, fg_color=PANEL_RAISED,
            corner_radius=10, text_color=ACCENT, height=28, width=118,
            font=("Consolas", 10),
        )
        self.scope_status.place(relx=1.0, x=-24, y=55, anchor="ne")
        self.scope_status.bind("<Button-1>", self._toggle_scope_session)
        self.scope_status.configure(cursor="hand2")

        controls_panel = ctk.CTkFrame(
            outer, fg_color=BG, bg_color=BG, border_width=1,
            border_color="#344B5D", corner_radius=18,
        )
        controls_panel.pack(fill="x", pady=(0, 13))
        controls_header = self._button(
            controls_panel, "Controls   ▾",
            lambda: self._toggle_section("Controls"), width=100,
        )
        controls_header.configure(
            fg_color="transparent", hover_color=PANEL_RAISED,
            anchor="w", height=46, font=("Segoe UI Semibold", 16),
        )
        controls_header.pack(fill="x", padx=10, pady=(7, 7))
        controls_body = tk.Frame(controls_panel, bg=BG)
        controls_body.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self.sections["Controls"] = {
            "header": controls_header, "subtitle": None,
            "body": controls_body, "collapsed": False,
            "body_pack": {"fill": "both", "expand": True,
                          "padx": 12, "pady": (4, 12)},
        }

        prompt_card, prompt_body = self._card(
            controls_body, "System prompt",
            "Exact text sent to Venus · changes apply to the next session",
        )
        prompt_card.pack(fill="x", pady=(0, 13))
        self.prompt = ctk.CTkTextbox(prompt_body, height=170, wrap="word",
                                     fg_color=PANEL_RAISED, text_color=TEXT,
                                     corner_radius=12, font=("Segoe UI", 11),
                                     border_width=0)
        self.prompt.pack(fill="x")
        self.prompt.insert("1.0", build_legacy_system_prompt(""))
        tk.Label(
            prompt_body,
            text="You can edit or remove every rule, including the prefix. Tool guidance must also be present here to affect the model.",
            bg=PANEL, fg=MUTED, font=("Segoe UI", 9), anchor="w",
        ).pack(fill="x", pady=(7, 0))
        self.prompt_save_status = tk.StringVar(value="Saved prompt is shown here at startup")
        prompt_footer = tk.Frame(prompt_body, bg=PANEL)
        prompt_footer.pack(fill="x", pady=(9, 0))
        tk.Label(prompt_footer, textvariable=self.prompt_save_status, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        self._button(prompt_footer, "Save prompt", self._save_prompt, width=120).pack(side="right")
        self._button(prompt_footer, "Add enabled tool rules", self._insert_tool_guidance,
                     width=165).pack(side="right", padx=(0, 8))
        self.prompt.bind("<KeyRelease>", self._prompt_changed)

        columns = ttk.Frame(controls_body, style="Dark.TFrame")
        columns.pack(fill="x")
        columns.columnconfigure(0, weight=1, uniform="halves")
        columns.columnconfigure(1, weight=1, uniform="halves")

        left = ttk.Frame(columns, style="Dark.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ttk.Frame(columns, style="Dark.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        card, devices = self._card(left, "Audio routing", "Choose the voices coming in and going out")
        card.pack(fill="x", pady=(0, 12))
        devices.columnconfigure(1, weight=1)
        ttk.Label(devices, text="MICROPHONE", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.input_box = ctk.CTkComboBox(devices, variable=self.vars["input_device"],
                                         state="readonly", corner_radius=10,
                                         fg_color=PANEL_RAISED, border_color="#385069",
                                         button_color="#304158", button_hover_color="#46627D",
                                         text_color=TEXT, dropdown_fg_color=PANEL_RAISED,
                                         dropdown_text_color=TEXT)
        self.input_box.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        ttk.Label(devices, text="SPEAKER", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.output_box = ctk.CTkComboBox(devices, variable=self.vars["output_device"],
                                          state="readonly", corner_radius=10,
                                          fg_color=PANEL_RAISED, border_color="#385069",
                                          button_color="#304158", button_hover_color="#46627D",
                                          text_color=TEXT, dropdown_fg_color=PANEL_RAISED,
                                          dropdown_text_color=TEXT)
        self.output_box.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
        self._button(devices, "Refresh devices", self._refresh_devices).grid(
            row=2, column=1, sticky="e", pady=(8, 4))
        ttk.Label(devices, textvariable=self.level_text, style="Muted.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 3))
        ctk.CTkProgressBar(devices, variable=self.level, progress_color=ACCENT,
                           fg_color=PANEL_RAISED, corner_radius=8, height=9).grid(
                               row=4, column=0, columnspan=2, sticky="ew")
        stt_switch = ctk.CTkCheckBox(
            devices, text="Transcribe mic in log  ⓘ", variable=self.vars["input_stt"],
            corner_radius=6, fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        stt_switch.grid(row=5, column=0, columnspan=2, sticky="w", pady=(13, 0))
        self._tooltip(
            stt_switch,
            "Run a small local English speech recognizer on the microphone and log only "
            "completed phrases. This is an independent estimate of the input; Venus "
            "does not expose its own transcript.",
        )
        echo_switch = ctk.CTkCheckBox(
            devices, text="Mute mic while Venus speaks  ⓘ",
            variable=self.vars["mute_mic_during_playback"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        echo_switch.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._tooltip(
            echo_switch,
            "Prevents Venus from hearing its own speaker output as a new question. "
            "You cannot interrupt Venus by voice while this is enabled. The mic "
            "resumes after playback finishes.",
        )
        vad_switch = ctk.CTkCheckBox(
            devices, text="Filter mic with Silero VAD  ⓘ",
            variable=self.vars["silero_vad"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        vad_switch.grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._tooltip(
            vad_switch,
            "Silences audio Silero classifies as non-speech, often including breathing. "
            "A short lead-in and tail preserve word boundaries. VAD alone cannot "
            "tell your voice from Venus's playback.",
        )
        aec_switch = ctk.CTkCheckBox(
            devices, text="Cancel Venus speaker echo  ⓘ",
            variable=self.vars["echo_cancellation"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        aec_switch.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._tooltip(
            aec_switch,
            "WebRTC echo cancellation uses the exact audio sent to the speaker "
            "to remove its echo from the microphone. The mic stays live, so you "
            "can interrupt Venus. Your input and output devices must support 48 kHz.",
        )
        self._entry(
            devices, 9, 0, "VAD threshold", "vad_threshold",
            "Speech detection threshold from 0.1 to 0.95. Higher values reject "
            "more noise but can miss quiet speech. 0.6 is the default.",
        )
        self._entry(devices, 10, 0, "Mic gain (dB)", "mic_gain_db",
                    PLAYBACK_HELP["mic_gain_db"])

        card, playback = self._card(left, "Playback", "Shape the pace and finish of every reply")
        card.pack(fill="x")
        playback.columnconfigure(1, weight=1)
        playback.columnconfigure(3, weight=1)
        sentence_switch = ctk.CTkCheckBox(
            playback, text="Wait for sentence punctuation  ⓘ",
            variable=self.vars["sentence_buffering"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        sentence_switch.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 9))
        self._tooltip(sentence_switch, PLAYBACK_HELP["sentence_buffering"])
        self._entry(playback, 1, 0, "Text tokens", "max_new_tokens_per_chunk",
                    PLAYBACK_HELP["max_new_tokens_per_chunk"])
        self._entry(playback, 1, 2, "Tail silence (s)", "playback_tail_seconds",
                    PLAYBACK_HELP["playback_tail_seconds"])
        self._entry(playback, 2, 0, "Speaker queue", "speaker_queue_size",
                    PLAYBACK_HELP["speaker_queue_size"])
        self._entry(playback, 2, 2, "TTS tokens", "tts_max_new_tokens_per_chunk",
                    PLAYBACK_HELP["tts_max_new_tokens_per_chunk"])
        self._entry(playback, 3, 0, "Final TTS min", "tts_final_min_tokens",
                    PLAYBACK_HELP["tts_final_min_tokens"])
        self._entry(playback, 3, 2, "Final TTS max", "tts_final_max_tokens",
                    PLAYBACK_HELP["tts_final_max_tokens"])
        debug_switch = ctk.CTkCheckBox(
            playback, text="Save reply audio for diagnosis  ⓘ",
            variable=self.vars["save_last_reply_debug"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        debug_switch.grid(row=4, column=0, columnspan=4, sticky="w", pady=(11, 0))
        self._tooltip(
            debug_switch,
            "Keeps realtime speaker playback and also saves each synthesized "
            "reply as a WAV under logs/. Compare a WAV "
            "with what you heard to locate a clipped ending.",
        )
        voice_label = ttk.Label(playback, text="VOICE WAV  ⓘ", style="Muted.TLabel",
                                cursor="question_arrow")
        voice_label.grid(row=5, column=0, sticky="w", pady=(11, 3))
        voice_entry = ctk.CTkEntry(playback, textvariable=self.vars["voice_reference"],
                                   corner_radius=10, fg_color=PANEL_RAISED,
                                   border_color="#385069", text_color=TEXT)
        voice_entry.grid(row=6, column=0, columnspan=3, sticky="ew", padx=(0, 8))
        voice_browse = self._button(playback, "Browse", self._browse_voice, width=95)
        voice_browse.grid(row=6, column=3, sticky="e")
        for widget in (voice_label, voice_entry, voice_browse):
            self._tooltip(widget, PLAYBACK_HELP["voice_reference"])
        ttk.Label(playback, text="Leave blank to use your project voice reference.",
                  style="Muted.TLabel").grid(row=7, column=0, columnspan=4, sticky="w", pady=(5, 0))
        self.playback_note = tk.StringVar(value="Settings apply when a session starts.")
        ttk.Label(playback, textvariable=self.playback_note, style="Muted.TLabel").grid(
            row=11, column=0, columnspan=4, sticky="w", pady=(9, 0))
        static_switch = ctk.CTkCheckBox(
            playback, text="Idle radio static  ⓘ", variable=self.vars["idle_static"],
            corner_radius=6, fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        static_switch.grid(row=9, column=0, columnspan=4, sticky="w", pady=(13, 0))
        self._tooltip(static_switch, PLAYBACK_HELP["idle_static"])
        squelch_row = tk.Frame(playback, bg=PANEL)
        squelch_row.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        squelch_row.columnconfigure(1, weight=1)
        squelch_label = ttk.Label(
            squelch_row, text="SQUELCH  ⓘ", style="Muted.TLabel", cursor="question_arrow",
        )
        squelch_label.grid(row=0, column=0, sticky="w")
        self.squelch_slider = ctk.CTkSlider(
            squelch_row, from_=-65, to=-25, number_of_steps=40,
            command=self._set_squelch, fg_color="#304158", progress_color=ACCENT,
            button_color=ACCENT, button_hover_color="#A5F1DA", height=18,
        )
        self.squelch_slider.grid(row=0, column=1, sticky="ew", padx=(12, 10))
        self.squelch_readout = tk.StringVar(value="-48 dBFS")
        squelch_value = tk.Label(
            squelch_row, textvariable=self.squelch_readout, bg=PANEL,
            fg=TEXT, font=("Consolas", 10), width=8, anchor="e",
        )
        squelch_value.grid(row=0, column=2, sticky="e")
        for widget in (squelch_label, self.squelch_slider, squelch_value):
            self._tooltip(widget, PLAYBACK_HELP["idle_static_level_dbfs"])

        card, generation = self._card(right, "Generation", "Balance responsiveness and consistency")
        card.pack(fill="x", pady=(0, 12))
        generation.columnconfigure(1, weight=1)
        generation.columnconfigure(3, weight=1)
        decode_label = ttk.Label(generation, text="DECODE MODE  ⓘ", style="Muted.TLabel",
                                 cursor="question_arrow")
        decode_label.grid(row=0, column=0, sticky="w", pady=5)
        decode_box = ctk.CTkComboBox(generation, variable=self.vars["decode_mode"],
                                     values=("sampling", "greedy"), state="readonly",
                                     corner_radius=10, width=100, fg_color=PANEL_RAISED,
                                     border_color="#385069", text_color=TEXT,
                                     dropdown_fg_color=PANEL_RAISED, dropdown_text_color=TEXT)
        decode_box.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=5)
        self._tooltip(decode_label, GENERATION_HELP["decode_mode"])
        self._tooltip(decode_box, GENERATION_HELP["decode_mode"])
        self._entry(generation, 0, 2, "Temperature", "temperature", GENERATION_HELP["temperature"])
        self._entry(generation, 1, 0, "Top K", "top_k", GENERATION_HELP["top_k"])
        self._entry(generation, 1, 2, "Top P", "top_p", GENERATION_HELP["top_p"])
        self._entry(generation, 2, 0, "Listen scale", "listen_prob_scale",
                    GENERATION_HELP["listen_prob_scale"])
        reply_switch = ctk.CTkCheckBox(
            generation, text="Favor reply after speech  ⓘ",
            variable=self.vars["reply_after_speech"], corner_radius=6,
            fg_color=ACCENT, hover_color="#A5F1DA", text_color=TEXT,
        )
        reply_switch.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self._tooltip(
            reply_switch,
            "When VAD hears speech and then silence, briefly reduce the model's "
            "chance of staying in listen. No text or synthetic user speech is added. "
            "This can make the model respond sooner to short requests.",
        )

        card, memory = self._card(right, "Conversation memory",
                                  "Keep recent exchanges as the session grows")
        card.pack(fill="x", pady=(0, 12))
        memory.columnconfigure(1, weight=1)
        self._entry(
            memory, 0, 0, "Keep last turns", "keep_last_turns",
            "Keep this many completed user-and-Venus exchanges in the model's "
            "conversation cache. Older exchanges are removed after each reply; "
            "the system prompt stays. Set 0 to disable. A safety cap can trim "
            "earlier if a few turns alone approach the 40,960-token limit.",
        )

        card, tools = self._card(right, "Optional tools", "Enable only the abilities you want to test")
        card.pack(fill="x")
        for row, (key, label, hint) in enumerate((
            ("calculator_tool", "Calculator", "Local arithmetic"),
            ("timer_tool", "Timer", "Local countdown alerts"),
            ("weather_tool", "Weather", "Current ZIP weather · internet"),
        )):
            ctk.CTkCheckBox(tools, text=label, variable=self.vars[key], corner_radius=6,
                            fg_color=ACCENT, hover_color="#A5F1DA",
                            text_color=TEXT).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(tools, text=hint, style="Muted.TLabel").grid(row=row, column=1, sticky="e", padx=(18, 0))
        tools.columnconfigure(1, weight=1)

        actions = tk.Frame(controls_body, bg=BG)
        actions.pack(fill="x", pady=(13, 12))
        self.start_button = self._button(actions, "●  Start listening", self._start,
                                          primary=True, width=170)
        self.start_button.pack(side="left")
        self.stop_button = self._button(actions, "Stop session", self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        tk.Label(actions, text="Runs locally on your RTX 5090", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="right")

        log_handle = tk.Frame(self.log_panel, bg=PANEL_RAISED, height=25,
                              cursor="sb_v_double_arrow")
        log_handle.pack(fill="x")
        log_handle.pack_propagate(False)
        handle_label = tk.Label(log_handle, text="↕  DRAG HERE TO RESIZE LOG  ↕",
                                bg=PANEL_RAISED, fg=TEXT,
                                font=("Segoe UI Semibold", 9), cursor="sb_v_double_arrow")
        handle_label.pack(expand=True)
        for widget in (log_handle, handle_label):
            widget.bind("<ButtonPress-1>", self._log_resize_start)
            widget.bind("<B1-Motion>", self._log_resize_move)
            widget.bind("<ButtonRelease-1>", self._log_resize_end)

        log_card, log_body = self._card(self.log_panel, "Session activity", "Live model output and audio diagnostics")
        log_card.pack(fill="both", expand=True)
        log_toolbar = ctk.CTkFrame(log_body, fg_color="transparent")
        log_toolbar.pack(fill="x", pady=(0, 8))
        tk.Label(log_toolbar, text="●  YOU", bg=PANEL, fg="#91C5FF",
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        tk.Label(log_toolbar, text="●  VENUS", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Semibold", 9)).pack(side="left", padx=(15, 0))
        self._button(log_toolbar, "Clear log", self._clear_log, width=100).pack(side="right")
        self.log = ctk.CTkTextbox(log_body, wrap="word", state="disabled", height=180,
                                  fg_color="#0E1621", text_color="#C8D7E7",
                                  corner_radius=12, font=("Consolas", 10),
                                  border_width=0)
        self.log.pack(fill="both", expand=True)
        self.log._textbox.tag_configure("muted", foreground="#7E94A9")
        self.log._textbox.tag_configure("good", foreground=ACCENT)
        self.log._textbox.tag_configure("input", foreground="#91C5FF")
        self.log._textbox.tag_configure("reply", foreground=ACCENT)
        self.log._textbox.tag_configure("warn", foreground=GOLD)
        self.log._textbox.tag_configure("error", foreground="#FF9A96")

    def _tooltip(self, widget, text):
        self.tooltips.append(HoverTip(widget, text))

    def _entry(self, parent, row, col, label, key, tooltip=None):
        name = ttk.Label(parent, text=label.upper() + ("  ⓘ" if tooltip else ""),
                         style="Muted.TLabel", cursor="question_arrow" if tooltip else "")
        name.grid(row=row, column=col, sticky="w", pady=7)
        entry = ctk.CTkEntry(parent, textvariable=self.vars[key], corner_radius=10,
                             fg_color=PANEL_RAISED, border_color="#385069",
                             text_color=TEXT, width=75 if tooltip else 95)
        entry.grid(row=row, column=col + 1, sticky="ew",
                   padx=(8, 8) if tooltip else (8, 14), pady=7)
        if tooltip:
            self._tooltip(name, tooltip)
            self._tooltip(entry, tooltip)

    def _clear_log(self):
        while True:
            try:
                self.log_items.get_nowait()
            except queue.Empty:
                break
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _browse_voice(self):
        path = filedialog.askopenfilename(title="Select voice reference", filetypes=[("WAV audio", "*.wav")])
        if path:
            self.vars["voice_reference"].set(path)

    def _refresh_devices(self):
        try:
            devices = sd.query_devices()
        except Exception as exc:
            messagebox.showerror("Audio devices", str(exc))
            return
        inputs = ["Default"]
        outputs = ["Default"]
        for index, device in enumerate(devices):
            label = f"{index}: {device['name']}"
            if device["max_input_channels"]:
                inputs.append(label)
            if device["max_output_channels"]:
                outputs.append(label)
        self.input_box.configure(values=inputs)
        self.output_box.configure(values=outputs)
        for key, values in (("input_device", inputs), ("output_device", outputs)):
            if self.vars[key].get() not in values:
                self.vars[key].set("Default")

    def _load_settings(self):
        saved = {}
        for path in (DEFAULT_SETTINGS, SETTINGS):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                saved.update(data)
        for key, var in self.vars.items():
            if key in saved:
                var.set(saved[key])
        if isinstance(saved.get("log_panel_height"), int):
            self.log_panel_height = max(180, min(1800, saved["log_panel_height"]))
        if isinstance(saved.get("system_prompt"), str):
            prompt_text = saved["system_prompt"]
            if saved.get("system_prompt_mode") != "full":
                prompt_text = build_legacy_system_prompt(
                    prompt_text,
                    bool(saved.get("calculator_tool")),
                    bool(saved.get("timer_tool")),
                    bool(saved.get("weather_tool")),
                )
            self.prompt.delete("1.0", "end")
            self.prompt.insert("1.0", prompt_text)
        for title in saved.get("collapsed_sections", []):
            if isinstance(title, str):
                self._set_section_collapsed(title, True)

    def _save_settings(self):
        data = {key: var.get() for key, var in self.vars.items()}
        data["system_prompt"] = self.prompt.get("1.0", "end").strip()
        data["system_prompt_mode"] = "full"
        data["collapsed_sections"] = [title for title, section in self.sections.items()
                                      if section["collapsed"]]
        if self.log_panel.winfo_height() > 1:
            self.log_panel_height = self.log_panel.winfo_height()
        data["log_panel_height"] = self.log_panel_height
        SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _restore_log_pane_height(self):
        height = self.log_pane.winfo_height()
        if height > 1:
            if self.sections.get("Controls", {}).get("collapsed"):
                desired = self.controls_outer.winfo_reqheight() + 4
                position = min(height - 180, max(250, desired))
            else:
                position = max(250, height - self.log_panel_height)
            self.log_pane.sash_place(0, 0, position)

    def _render_wallpaper(self, event):
        size = (event.width, event.height)
        if self.wallpaper_source is None or min(size) < 2 or size == self.wallpaper_size:
            return
        self.wallpaper_size = size
        image = ImageOps.fit(
            self.wallpaper_source, size,
            method=Image.Resampling.LANCZOS, centering=(0.5, 0.45),
        )
        image = Image.blend(image, Image.new("RGB", size, BG), 0.58)
        self.wallpaper_photo = ImageTk.PhotoImage(image)
        self.wallpaper_label.configure(image=self.wallpaper_photo)

    def _log_resize_start(self, event):
        self._log_drag_origin = (event.y_root, self.log_pane.sash_coord(0)[1])

    def _log_resize_move(self, event):
        if self._log_drag_origin is None:
            return
        start_y, sash_y = self._log_drag_origin
        self.log_pane.sash_place(0, 0, sash_y + event.y_root - start_y)

    def _log_resize_end(self, _event):
        self._log_drag_origin = None
        try:
            self._save_settings()
        except OSError:
            pass

    def _prompt_changed(self, _event=None):
        self.prompt_save_status.set("Unsaved changes · applies to the next session")

    def _playback_setting_changed(self, *_args):
        if self.proc is not None and self.proc.poll() is None:
            self.playback_note.set("Changed · stop and start to apply these settings.")

    def _set_squelch(self, value):
        level = int(round(value))
        self.vars["idle_static_level_dbfs"].set(str(level))
        self.squelch_readout.set(f"{level} dBFS")

    def _sync_squelch_readout(self):
        try:
            level = float(self.vars["idle_static_level_dbfs"].get())
        except ValueError:
            level = -48.0
        level = max(-65.0, min(-25.0, level))
        self.squelch_slider.set(level)
        self.squelch_readout.set(f"{level:.0f} dBFS")

    def _schedule_idle_sync(self, *_args):
        if self._idle_sync_job is not None:
            self.root.after_cancel(self._idle_sync_job)
        self._idle_sync_job = self.root.after(350, self._finish_idle_sync)

    def _finish_idle_sync(self):
        self._idle_sync_job = None
        self._sync_idle_player()

    def _sync_idle_player(self):
        if self.closing or (self.proc is not None and self.proc.poll() is None):
            self.idle_player.stop()
            return
        if not self.vars["idle_static"].get():
            self.idle_player.stop()
            return
        try:
            level = float(self.vars["idle_static_level_dbfs"].get())
            if not -65 <= level <= -25:
                return
            selected = self.vars["output_device"].get()
            device = None if selected == "Default" else int(selected.split(":", 1)[0])
        except ValueError:
            return
        self.idle_player.start(device, level)

    def _save_prompt(self):
        try:
            self._save_settings()
        except OSError as exc:
            messagebox.showerror("Could not save prompt", str(exc))
            return
        self.prompt_save_status.set("Saved · will load at next startup")

    def _insert_tool_guidance(self):
        enabled = (
            ("calculator_tool", CALCULATOR_INSTRUCTIONS),
            ("timer_tool", TIMER_INSTRUCTIONS),
            ("weather_tool", WEATHER_INSTRUCTIONS),
        )
        current = self.prompt.get("1.0", "end").strip()
        additions = [rule for key, rule in enabled if self.vars[key].get() and rule not in current]
        if not additions:
            self.prompt_save_status.set("No new enabled tool rules to add")
            return
        self.prompt.insert("end", "\n\n" + "\n\n".join(additions))
        self.prompt_save_status.set("Tool rules added · save and restart to apply")

    def _command(self):
        values = self.vars
        numeric = (
            ("max_new_tokens_per_chunk", int, 2, 256),
            ("tts_max_new_tokens_per_chunk", int, 26, 128),
            ("tts_final_min_tokens", int, 0, 127),
            ("tts_final_max_tokens", int, 26, 256),
            ("playback_tail_seconds", float, 0, 2),
            ("idle_static_level_dbfs", float, -65, -25),
            ("mic_gain_db", float, -24, 12),
            ("speaker_queue_size", int, 1, 64),
            ("temperature", float, 0.01, 2),
            ("top_k", int, 1, 200),
            ("top_p", float, 0.01, 1),
            ("listen_prob_scale", float, 0, 5),
            ("vad_threshold", float, 0.1, 0.95),
            ("keep_last_turns", int, 0, 50),
        )
        command = [sys.executable, "-u", str(SCRIPT), "--control-stdin"]
        for key, converter, low, high in numeric:
            raw = values[key].get().strip()
            try:
                number = converter(raw)
            except ValueError as exc:
                raise ValueError(f"{key.replace('_', ' ')} needs a number") from exc
            if not low <= number <= high:
                raise ValueError(f"{key.replace('_', ' ')} must be between {low} and {high}")
            command += ["--" + key.replace("_", "-"), raw]
        for key in ("input_device", "output_device"):
            selection = values[key].get()
            if selection != "Default":
                command += ["--" + key.replace("_", "-"), selection.split(":", 1)[0]]
        if values["voice_reference"].get().strip():
            command += ["--voice-reference", values["voice_reference"].get().strip()]
        if int(values["tts_final_min_tokens"].get()) >= int(values["tts_final_max_tokens"].get()):
            raise ValueError("Final TTS min must be lower than Final TTS max")
        if not values["input_stt"].get():
            command.append("--no-input-stt")
        if not values["mute_mic_during_playback"].get():
            command.append("--no-mute-mic-during-playback")
        if values["echo_cancellation"].get():
            command.append("--echo-cancellation")
        if values["silero_vad"].get():
            command.append("--silero-vad")
        if not values["reply_after_speech"].get():
            command.append("--no-reply-after-speech")
        if values["save_last_reply_debug"].get():
            command.append("--save-last-reply-debug")
        if values["sentence_buffering"].get():
            command.append("--sentence-buffering")
        if not values["idle_static"].get():
            command.append("--no-idle-static")
        for key in ("calculator_tool", "timer_tool", "weather_tool"):
            if values[key].get():
                command.append("--" + key.replace("_", "-"))
        command += ["--decode-mode", values["decode_mode"].get()]
        system_prompt = self.prompt.get("1.0", "end").strip()
        if not system_prompt:
            raise ValueError("System prompt must contain text")
        command += ["--exact-system-prompt", "--system-prompt", system_prompt]
        return command

    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        log_handle = None
        try:
            command = self._command()
            self._save_settings()
            self.prompt_save_status.set("Saved · will load at next startup")
            log_dir = PROJECT / "logs"
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f"venus-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            self.idle_player.stop()
            self.proc = subprocess.Popen(
                command, cwd=PROJECT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except (OSError, ValueError) as exc:
            if log_handle is not None:
                log_handle.close()
            self._sync_idle_player()
            messagebox.showerror("Could not start", str(exc))
            return
        self.status.set("Loading model…")
        self._sync_scope_action()
        self.playback_note.set("Active values appear in Session activity. Restart after changes.")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._append_log(f"\n--- Starting live session ---\nLog: {log_path}\n")
        threading.Thread(target=self._read_output, args=(self.proc, log_handle), daemon=True).start()

    def _read_output(self, proc, log_handle):
        try:
            for line in proc.stdout:
                if log_handle is not None and not line.startswith(("[meter]", "[scope]")):
                    try:
                        log_handle.write(line)
                        log_handle.flush()
                    except OSError as exc:
                        self.log_items.put(f"[log] Could not write session log: {exc}\n")
                        log_handle.close()
                        log_handle = None
                self.log_items.put(line)
        finally:
            if log_handle is not None:
                log_handle.close()
            self.log_items.put(None)

    def _stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.status.set("Stopping and draining audio…")
        self._sync_scope_action()
        self.stop_button.configure(state="disabled")
        try:
            self.proc.stdin.write("stop\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self.proc.terminate()

    def _toggle_scope_session(self, _event=None):
        if self.closing or self.status.get().startswith("Stopping"):
            return
        if self.proc is not None and self.proc.poll() is None:
            self._stop()
        else:
            self._start()

    def _sync_scope_action(self):
        if self.status.get().startswith("Stopping"):
            self.scope.set_action("STOPPING…", enabled=False)
        elif self.proc is not None and self.proc.poll() is None:
            self.scope.set_action("■  STOP SESSION")
        else:
            self.scope.set_action("▶  START SESSION")

    def _append_log(self, value):
        lower = value.lower()
        if lower.startswith("[input audio] ") or (lower.startswith("[input stt] ")
                and "local diagnostic transcript enabled" not in lower
                and "dropped " not in lower):
            tag = "input"
        elif lower.startswith("[turn complete]") or lower.startswith("[spoken sentence]"):
            tag = "reply"
        elif "traceback" in lower or "error" in lower:
            tag = "error"
        elif "[latency]" in lower or "[playback]" in lower or "failed" in lower:
            tag = "warn"
        elif "[tool:" in lower or "Listening. Press Ctrl+C" in value:
            tag = "good"
        elif "[audio chunk" in lower or "[listening]" in lower:
            tag = "muted"
        else:
            tag = None
        self.log.configure(state="normal")
        start = self.log.index("end-1c")
        self.log.insert("end", value, tag)
        if lower.startswith("[audio chunk") and "state=speak" in lower:
            match = re.search(r"\btext=(.+?) audio=", value)
            if match and match.group(1) not in ("''", '""'):
                self.log._textbox.tag_add(
                    "reply", f"{start}+{match.start(1)}c", f"{start}+{match.end(1)}c"
                )
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "500.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        for _ in range(200):
            try:
                item = self.log_items.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                mic_level = re.search(r"mic=(-?\d+)dBFS", item)
                if mic_level:
                    dbfs = int(mic_level.group(1))
                    self.level.set(max(0, min(1, (dbfs + 80) / 70)))
                    self.level_text.set(f"MIC  ·  {dbfs} dBFS")
                if item.startswith("[meter]"):
                    continue
                if item.startswith("[scope] "):
                    try:
                        rms, packed = item[8:].strip().split(" ", 1)
                        self.scope.set_signal(float(rms), [int(v) for v in packed.split(",")])
                    except ValueError:
                        pass
                    continue
                self._append_log(item)
                if "Listening. Press Ctrl+C" in item:
                    self.status.set("Listening")
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.proc = None
            self.status.set(f"Stopped (exit {code})")
            self.level.set(0)
            self.level_text.set("MIC  ·  — dBFS")
            self.scope.clear()
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if not self.closing:
                self._sync_idle_player()
        self._sync_scope_action()
        if self.closing and self.proc is None:
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def _close(self):
        if self._idle_sync_job is not None:
            self.root.after_cancel(self._idle_sync_job)
            self._idle_sync_job = None
        self.idle_player.stop()
        try:
            self._save_settings()
        except OSError:
            pass
        if self.proc is None or self.proc.poll() is not None:
            self.root.destroy()
            return
        if self.closing:
            self.proc.terminate()
            return
        self.closing = True
        self._stop()


if __name__ == "__main__":
    root = ctk.CTk(fg_color=BG)
    LiveMicUI(root)
    root.mainloop()
