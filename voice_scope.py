"""Speaker-driven synthetic voice display for the local console UI."""

import math
import time
import tkinter as tk

from PIL import Image, ImageOps, ImageTk


class VoiceScope(tk.Canvas):
    BACKGROUND = "#07141F"
    GRID = "#183044"
    MUTED = "#68879B"
    CYAN = "#76F1DD"
    AMBER = "#F8B86D"

    def __init__(self, parent, on_toggle=None, backdrop_path=None):
        super().__init__(parent, height=216, bg=self.BACKGROUND,
                         highlightthickness=0, borderwidth=0, takefocus=1)
        self.on_toggle = on_toggle
        self.backdrop_source = (
            Image.open(backdrop_path).convert("RGB")
            if backdrop_path is not None and backdrop_path.is_file() else None
        )
        self.backdrop_photo = None
        self.action_label = "START SESSION"
        self.action_enabled = True
        self.hovered = False
        self.target = [0.0] * 48
        self.wave = [0.0] * 48
        self.level = 0.0
        self.target_level = 0.0
        self.last_signal = 0.0
        self.phase = 0.0
        self.bind("<Configure>", self._draw_grid)
        self.bind("<Button-1>", self._activate)
        self.bind("<Return>", self._activate)
        self.bind("<space>", self._activate)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.after(40, self._animate)

    def set_action(self, label, enabled=True):
        self.action_label = label
        self.action_enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")

    def _activate(self, _event=None):
        if self.action_enabled and self.on_toggle is not None:
            self.focus_set()
            self.on_toggle()

    def _enter(self, _event=None):
        self.hovered = True

    def _leave(self, _event=None):
        self.hovered = False

    def set_signal(self, rms, peaks):
        if len(peaks) != 48:
            return
        self.target = [max(-100, min(100, value)) for value in peaks]
        self.target_level = min(1.0, max(0.0, rms * 9.0))
        self.last_signal = time.monotonic()

    def clear(self):
        self.target = [0.0] * 48
        self.target_level = 0.0
        self.last_signal = 0.0

    def _draw_grid(self, _event=None):
        self.delete("static")
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 50:
            return
        if self.backdrop_source is not None:
            backdrop = ImageOps.fit(
                self.backdrop_source, (width, height),
                method=Image.Resampling.LANCZOS, centering=(0.5, 0.28),
            )
            backdrop = Image.blend(
                backdrop, Image.new("RGB", (width, height), self.BACKGROUND), 0.73,
            )
            self.backdrop_photo = ImageTk.PhotoImage(backdrop)
            self.create_image(0, 0, anchor="nw", image=self.backdrop_photo,
                              tags="static")
        for x in range(16, width, 32):
            self.create_line(x, 10, x, height - 11, fill=self.GRID,
                             dash=(2, 6), tags="static")
        for y in range(16, height, 32):
            self.create_line(10, y, width - 11, y, fill=self.GRID,
                             dash=(2, 6), tags="static")
        middle = height / 2
        self.create_line(18, middle, width - 18, middle, fill="#31576A",
                         dash=(4, 5), tags="static")
        for radius in (36, 62, 89):
            self.create_oval(width / 2 - radius, middle - radius,
                             width / 2 + radius, middle + radius,
                             outline="#1C3B4E", tags="static")
        self.create_line(width / 2, 17, width / 2, height - 17,
                         fill="#31576A", dash=(3, 5), tags="static")
        for x in range(20, width - 16, 16):
            length = 7 if (x - 20) % 64 == 0 else 3
            self.create_line(x, height - 10, x, height - 10 - length,
                             fill="#3B6878", tags="static")
        for x, direction in ((10, 1), (width - 10, -1)):
            for y, vertical in ((10, 1), (height - 10, -1)):
                self.create_line(x, y + 13 * vertical, x, y,
                                 x + 13 * direction, y, fill=self.CYAN,
                                 width=2, tags="static")
        self.create_text(27, 25, text="CH 01  /  AUDIO OUTPUT", anchor="w",
                         fill=self.MUTED, font=("Consolas", 10), tags="static")
        self.create_text(width - 26, height - 25, text="24.0 kHz  •  MONO",
                         anchor="e", fill=self.MUTED,
                         font=("Consolas", 9), tags="static")
        self.tag_lower("static")

    def _animate(self):
        if time.monotonic() - self.last_signal > 0.27:
            self.target_level = 0.0
            self.target = [0.0] * 48
        self.level += (self.target_level - self.level) * 0.28
        self.wave = [current + (target - current) * 0.37
                     for current, target in zip(self.wave, self.target)]
        self.phase += 0.22 if self.level > 0.015 else 0.025
        self._draw_signal()
        self.after(40, self._animate)

    def _draw_signal(self):
        self.delete("dynamic")
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 100:
            return
        middle = height / 2
        active = self.level > 0.025
        left, right = 31, width - 31
        points = []
        reflection = []
        for index, sample in enumerate(self.wave):
            x = left + (right - left) * index / 47
            taper = 0.55 + 0.45 * math.sin(math.pi * index / 47)
            motion = math.sin(index * 0.52 + self.phase) * self.level * 14
            displacement = max(-65, min(65, sample * (1.45 + self.level) * taper + motion))
            points.extend((x, middle - displacement))
            reflection.extend((x, middle + displacement * 0.30))
        self.create_line(*reflection, fill="#1F736E", width=2,
                         smooth=True, splinesteps=10, tags="dynamic")
        self.create_line(*points, fill="#165D62", width=14,
                         smooth=True, splinesteps=10, tags="dynamic")
        self.create_line(*points, fill="#2DAEAA", width=7,
                         smooth=True, splinesteps=10, tags="dynamic")
        self.create_line(*points, fill=self.CYAN, width=2,
                         smooth=True, splinesteps=10, tags="dynamic")
        # Segmented power rail reads like a console meter without becoming a label.
        for index in range(22):
            x = 28 + index * 9
            bright = index < round(self.level * 22)
            self.create_rectangle(x, height - 30, x + 5, height - 24,
                                  fill=self.AMBER if bright else "#244454",
                                  outline="", tags="dynamic")
        self.create_oval(width - 139, 18, width - 131, 26,
                         fill=self.AMBER if active else "#426175",
                         outline="", tags="dynamic")
        self.create_text(width - 119, 22,
                         text="VOICE ACTIVE" if active else "STANDBY",
                         anchor="w", fill=self.AMBER if active else self.MUTED,
                         font=("Consolas", 10), tags="dynamic")
        # The entire display is a control; the pill makes that action discoverable.
        button_width = 198
        button_height = 29
        x0 = width / 2 - button_width / 2
        x1 = width / 2 + button_width / 2
        y0 = height - 43
        y1 = y0 + button_height
        fill = ("#315866" if self.hovered else "#1B3545") if self.action_enabled else "#172B38"
        radius = button_height / 2
        self.create_oval(x0, y0, x0 + button_height, y1,
                         fill=fill, outline="", tags="dynamic")
        self.create_oval(x1 - button_height, y0, x1, y1,
                         fill=fill, outline="", tags="dynamic")
        self.create_rectangle(x0 + radius, y0, x1 - radius, y1,
                              fill=fill, outline="", tags="dynamic")
        self.create_text(width / 2, y0 + radius, text=self.action_label,
                         fill=self.CYAN if self.action_enabled else self.MUTED,
                         font=("Consolas", 10, "bold"), tags="dynamic")
