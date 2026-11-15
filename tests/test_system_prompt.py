"""The UI must send exactly the text shown in its prompt editor."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from live_mic_duplex import (
    CALCULATOR_INSTRUCTIONS, build_legacy_system_prompt, resolve_system_prompt,
)


class SystemPromptTests(unittest.TestCase):
    def test_legacy_migration_matches_known_working_prompt(self):
        user_text = (
            "Speak English. When asked for a story, tell a complete story in five to eight "
            "short sentences. Begin in action. Use active voice, concrete details, crisp "
            "dialogue, and little exposition. Keep most sentences under twelve words. "
            "End on a sharp image or sudden turn."
        )
        path = Path(__file__).resolve().parents[1] / "logs" / "venus-20261114-140453-275588.log"
        if not path.exists():
            self.skipTest("Local comparison log is unavailable")
        logged = next(
            line.removeprefix("Effective system prompt: ")
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("Effective system prompt: ")
        )
        self.assertEqual(build_legacy_system_prompt(user_text), logged)

    def test_exact_prompt_ignores_all_automatic_rules(self):
        prompt = "Custom prefix. Speak plainly. \u201cLiteral quotes\u201d"
        args = SimpleNamespace(
            exact_system_prompt=True, system_prompt=prompt,
            calculator_tool=True, timer_tool=True, weather_tool=True,
        )
        self.assertEqual(resolve_system_prompt(args), prompt)

    def test_old_cli_still_adds_enabled_tool_rules(self):
        args = SimpleNamespace(
            exact_system_prompt=False, system_prompt="Answer briefly.",
            calculator_tool=True, timer_tool=False, weather_tool=False,
        )
        self.assertIn(CALCULATOR_INSTRUCTIONS, resolve_system_prompt(args))


if __name__ == "__main__":
    unittest.main()
