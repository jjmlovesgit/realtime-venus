"""Keep recent completed duplex exchanges in the language-model KV cache."""

from collections import deque


DEFAULT_HIGH_TOKENS = 36_000
DEFAULT_LOW_TOKENS = 30_000


class TurnRetention:
    """Prune whole one-second units grouped by completed assistant turns.

    Realtime-Venus tracks cache units, not dialogue turns. A completed reply
    closes an exchange containing all units since the previous reply. The
    checkpoint's own cache-drop method preserves the system prompt and
    realigns rotary positions after eviction.
    """

    def __init__(self, keep_turns=3, high_tokens=DEFAULT_HIGH_TOKENS,
                 low_tokens=DEFAULT_LOW_TOKENS):
        if keep_turns < 1:
            raise ValueError("keep_turns must be positive")
        if not 0 < low_tokens < high_tokens:
            raise ValueError("context limits must satisfy 0 < low < high")
        self.keep_turns = keep_turns
        self.high_tokens = high_tokens
        self.low_tokens = low_tokens
        self.completed_ends = deque()

    @staticmethod
    def _drop_through(decoder, cutoff_unit_id):
        history = decoder._unit_history
        count = 0
        removed_tokens = 0
        for entry in history:
            if entry["unit_id"] > cutoff_unit_id:
                break
            count += 1
            removed_tokens += entry["length"]
        if not count:
            return None
        cache_before = decoder.get_cache_length()
        expected = decoder._system_preserve_length + sum(entry["length"] for entry in history)
        if cache_before != expected:
            raise RuntimeError(
                f"Context cache accounting mismatch: cache={cache_before}, tracked={expected}"
            )
        if not decoder._drop_tokens_from_cache(removed_tokens):
            raise RuntimeError("Could not evict old context from the model cache")
        del history[:count]
        cache_after = decoder.get_cache_length()
        if cache_after != cache_before - removed_tokens:
            raise RuntimeError("Context cache length did not decrease as expected")
        decoder._sliding_event_count += 1
        decoder._total_dropped_tokens += removed_tokens
        decoder._total_dropped_units += count
        return {"units": count, "tokens": removed_tokens, "cache": cache_after}

    def completed_turn(self, decoder):
        """Mark a spoken reply complete; discard exchanges older than N."""
        history = decoder._unit_history
        if not history:
            return None
        self.completed_ends.append(history[-1]["unit_id"])
        if len(self.completed_ends) <= self.keep_turns:
            return None
        oldest_end = self.completed_ends.popleft()
        result = self._drop_through(decoder, oldest_end)
        if result:
            result["reason"] = "old turn"
        return result

    def enforce_safety_limit(self, decoder):
        """Leave room below the checkpoint's context limit, even in long turns."""
        if decoder.get_cache_length() <= self.high_tokens:
            return []
        events = []
        while decoder.get_cache_length() > self.low_tokens and decoder._unit_history:
            if self.completed_ends:
                cutoff = self.completed_ends.popleft()
                reason = "context limit; completed turn"
            else:
                cutoff = decoder._unit_history[0]["unit_id"]
                reason = "context limit; partial turn"
            result = self._drop_through(decoder, cutoff)
            if result is None:
                break
            result["reason"] = reason
            events.append(result)
        if decoder.get_cache_length() > self.high_tokens:
            raise RuntimeError("Context could not be reduced below the safety limit")
        return events
