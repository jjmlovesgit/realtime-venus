import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from context_retention import TurnRetention


class FakeDecoder:
    def __init__(self, lengths, system_length=20):
        self._system_preserve_length = system_length
        self._unit_history = [
            {"unit_id": index, "length": length}
            for index, length in enumerate(lengths, 1)
        ]
        self.cache_length = system_length + sum(lengths)
        self._sliding_event_count = 0
        self._total_dropped_tokens = 0
        self._total_dropped_units = 0

    def get_cache_length(self):
        return self.cache_length

    def _drop_tokens_from_cache(self, length):
        if length > self.cache_length - self._system_preserve_length:
            return False
        self.cache_length -= length
        return True


class TurnRetentionTests(unittest.TestCase):
    def test_real_decoder_cache_preserves_prompt(self):
        source = Path(__file__).resolve().parents[1] / "Realtime-Venus-Audio" / "utils.py"
        spec = importlib.util.spec_from_file_location("venus_context_test_utils", source)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        class Tokenizer:
            eos_token_id = 0
            unk_token_id = -1
            all_special_ids = []
            all_special_tokens = []

            def convert_tokens_to_ids(self, _token):
                return 1

        decoder = module.StreamDecoder(
            SimpleNamespace(config=SimpleNamespace(rope_theta=10000.0)), Tokenizer(),
        )
        cache = DynamicCache()
        original = torch.arange(40, dtype=torch.float32).reshape(1, 1, 20, 2)
        cache.update(original, original.clone(), 0)
        decoder.cache = cache
        decoder._system_preserve_length = 2
        decoder._unit_history = [
            {"unit_id": index, "length": length}
            for index, length in enumerate((3, 4, 5, 6), 1)
        ]
        retention = TurnRetention(keep_turns=3, high_tokens=100, low_tokens=80)
        retention.completed_ends.extend((1, 2, 3))
        event = retention.completed_turn(decoder)
        self.assertEqual(event["tokens"], 3)
        self.assertEqual(decoder.get_cache_length(), 17)
        self.assertTrue(torch.equal(decoder.cache.key_cache[0][:, :, :2, :],
                                    original[:, :, :2, :]))

    def test_keeps_three_completed_exchanges_and_system_prompt(self):
        decoder = FakeDecoder([10, 15])
        retention = TurnRetention(keep_turns=3, high_tokens=500, low_tokens=300)
        self.assertIsNone(retention.completed_turn(decoder))
        for end_id, length in ((3, 20), (4, 25)):
            decoder._unit_history.append({"unit_id": end_id, "length": length})
            decoder.cache_length += length
            self.assertIsNone(retention.completed_turn(decoder))
        decoder._unit_history.append({"unit_id": 5, "length": 30})
        decoder.cache_length += 30
        event = retention.completed_turn(decoder)
        self.assertEqual(event["units"], 2)
        self.assertEqual([unit["unit_id"] for unit in decoder._unit_history], [3, 4, 5])
        self.assertEqual(decoder.get_cache_length(), 20 + 20 + 25 + 30)
        self.assertEqual(list(retention.completed_ends), [3, 4, 5])

    def test_safety_cap_can_drop_older_turn_early(self):
        decoder = FakeDecoder([100, 100, 100, 100], system_length=20)
        retention = TurnRetention(keep_turns=3, high_tokens=350, low_tokens=250)
        retention.completed_ends.extend([1, 2])
        events = retention.enforce_safety_limit(decoder)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["reason"], "context limit; completed turn")
        self.assertEqual(decoder.get_cache_length(), 220)
        self.assertEqual([unit["unit_id"] for unit in decoder._unit_history], [3, 4])


if __name__ == "__main__":
    unittest.main()
