"""Small local tools for the live Realtime-Venus microphone session."""

from __future__ import annotations

import ast
import json
import math
import operator
import queue
import re
import threading
import time


class DelegateParser:
    """Collect native Realtime-Venus delegate spans from generated token IDs."""

    _CONTROL_TOKENS = (
        "<|speak|>", "<|listen|>", "<|turn_eos|>", "<|chunk_eos|>",
        "<unit>", "</unit>", "<image>", "</image>", "<delegate>",
        "</delegate>", "<backend>", "</backend>",
    )

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in self._CONTROL_TOKENS
        }
        if any(value is None or value < 0 for value in self.token_ids.values()):
            raise RuntimeError("Tokenizer is missing Realtime-Venus delegate protocol tokens.")
        if len(set(self.token_ids.values())) != len(self.token_ids):
            raise RuntimeError("Tokenizer has colliding Realtime-Venus protocol token IDs.")
        self.ignore_ids = set(self.token_ids.values()) - {
            self.token_ids["<delegate>"], self.token_ids["</delegate>"]
        }
        self.active = False
        self.body_ids: list[int] = []

    def consume(self, token_ids) -> tuple[bool, bool, str | None]:
        """Return delegate presence, whether it remains open, and any completed body."""
        touched = self.active
        completed = None
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id == self.token_ids["<delegate>"]:
                touched = True
                self.active = True
                self.body_ids.clear()
            elif self.active and token_id in {
                self.token_ids["</delegate>"], self.token_ids["<|turn_eos|>"]
            }:
                completed = self.tokenizer.decode(
                    self.body_ids, skip_special_tokens=True
                ).strip()
                self.body_ids.clear()
                self.active = False
                touched = True
            elif self.active and token_id not in self.ignore_ids:
                self.body_ids.append(token_id)
        return touched, self.active, completed


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_CALCULATOR_VALUE = 1e100
_EXPLICIT_TIMER_START = re.compile(
    r"\b(?:set|start|create|begin|run)\b.{0,50}\b(?:timer|countdown)\b"
    r"|\b(?:timer|countdown)\b.{0,40}\b(?:for|to|start|set)\b",
    re.IGNORECASE,
)
_EXPLICIT_TIMER_CANCEL = re.compile(
    r"\b(?:cancel|stop|clear)\b.{0,40}\b(?:timer|countdown)\b"
    r"|\b(?:timer|countdown)\b.{0,40}\b(?:cancel|stop)\b",
    re.IGNORECASE,
)


def calculate(expression: str) -> str:
    """Evaluate bounded arithmetic syntax without eval or access to Python names."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 160:
        raise ValueError("Enter a short arithmetic expression.")
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ValueError("I couldn't parse that calculation.") from exc
    if sum(1 for _ in ast.walk(root)) > 40:
        raise ValueError("That calculation is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponents must be between -100 and 100.")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        else:
            raise ValueError("Use numbers, parentheses, and +, -, *, /, //, %, or ** only.")
        if not math.isfinite(value) or abs(value) > _MAX_CALCULATOR_VALUE:
            raise ValueError("That calculation is outside the supported range.")
        return value

    try:
        result = evaluate(root)
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError("That calculation is undefined.") from exc
    if result.is_integer():
        return str(int(result))
    return f"{result:.10g}"


class TimerManager:
    """Manage local countdown timers and enqueue spoken expiration notices."""

    MAX_SECONDS = 24 * 60 * 60

    def __init__(self, expired_events: queue.Queue):
        self.expired_events = expired_events
        self._lock = threading.Lock()
        self._timers: dict[str, tuple[threading.Timer, str, float]] = {}
        self._next_id = 1

    def start(self, seconds, label="timer") -> str:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("Timer duration must be a number of seconds.")
        duration = float(seconds)
        if not math.isfinite(duration) or not 1 <= duration <= self.MAX_SECONDS:
            raise ValueError("Timer duration must be between 1 second and 24 hours.")
        if not isinstance(label, str):
            raise ValueError("Timer label must be text.")
        label = " ".join(label.split())[:80] or "timer"

        with self._lock:
            timer_id = f"T{self._next_id}"
            self._next_id += 1
            deadline = time.monotonic() + duration
            timer = threading.Timer(duration, self._expire, args=(timer_id,))
            timer.daemon = True
            self._timers[timer_id] = (timer, label, deadline)
            timer.start()
        return f"Timer {timer_id} started for {self._format_duration(duration)}."

    def cancel(self, timer_id: str) -> str:
        if not isinstance(timer_id, str):
            raise ValueError("Specify the timer ID, such as T1.")
        key = timer_id.strip().upper()
        with self._lock:
            record = self._timers.pop(key, None)
        if record is None:
            return f"I couldn't find an active timer named {key}."
        record[0].cancel()
        return f"Timer {key} canceled."

    def close(self):
        with self._lock:
            timers = [record[0] for record in self._timers.values()]
            self._timers.clear()
        for timer in timers:
            timer.cancel()

    def _expire(self, timer_id: str):
        with self._lock:
            record = self._timers.pop(timer_id, None)
        if record is not None:
            label = record[1]
            if label.casefold() in {"timer", "countdown"}:
                notice = f"Your timer ({timer_id}) is up."
            else:
                notice = f"Your {label} timer ({timer_id}) is up."
            self.expired_events.put(
                notice
            )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = round(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
        if minutes:
            parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
        if secs or not parts:
            parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
        return " ".join(parts)


def execute_local_tool(request_text: str, enabled_tools: set[str], timers: TimerManager) -> str:
    """Validate a delegate JSON request and run an enabled local tool."""
    try:
        request = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool request was not valid JSON.") from exc
    if not isinstance(request, dict):
        raise ValueError("Tool request must be a JSON object.")
    tool = request.get("tool")
    if tool not in enabled_tools:
        raise ValueError("That tool is not enabled in this session.")
    if tool == "calculator":
        result = calculate(request.get("expression"))
        return f"The answer is {result}."
    if tool == "timer_start":
        user_request = request.get("user_request")
        if not isinstance(user_request, str) or not _EXPLICIT_TIMER_START.search(user_request):
            raise ValueError("Timer not started: no clear timer-start request was provided.")
        return timers.start(request.get("duration_seconds"), request.get("label", "timer"))
    if tool == "timer_cancel":
        user_request = request.get("user_request")
        if not isinstance(user_request, str) or not _EXPLICIT_TIMER_CANCEL.search(user_request):
            raise ValueError("Timer not canceled: no clear timer-cancel request was provided.")
        return timers.cancel(request.get("timer_id"))
    raise ValueError("Unknown local tool.")
