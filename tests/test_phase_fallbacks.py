"""Tests for phase fallback behavior and extraction helpers.

Every phase falls back to a degraded template when the LLM call fails or is
unconfigured; the fallback text is what users actually see in degraded mode,
so its structure (and the regex extraction helpers) is pinned here.
"""

import unittest

from agent.profile_builder import ProfileBuilder
from agent.report_generator import ReportGenerator
from agent.synthesizer import Synthesizer
from agent.validator_team import ValidatorTeam

from tests.helpers import failure_response, success_response


class ValidatorTeamTest(unittest.TestCase):
    def test_fallback_flags_missing_client_and_round(self):
        text = ValidatorTeam()._validate_fallback("t", "p", 2)
        self.assertIn("No LLM client configured", text)
        self.assertIn("Round: 2", text)

    def test_llm_failure_falls_back(self):
        validator = ValidatorTeam()
        text = validator._validate_with_llm(
            "t", "p", 1, FakeFailingLLM()
        )
        self.assertIn("No LLM client configured", text)

    def test_llm_success_returns_content(self):
        lll = ScriptedLLM([success_response("VALIDATION OUTPUT")])
        text = ValidatorTeam().validate("t", "p", 1, lll)
        self.assertEqual(text, "VALIDATION OUTPUT")


class SynthesizerTest(unittest.TestCase):
    def test_fallback_appends_improvements(self):
        text = Synthesizer()._merge_fallback(
            "t", "ORIGINAL PROFILE", ["Add SEC revenue data"], 3
        )
        self.assertIn("ORIGINAL PROFILE", text)
        self.assertIn("ADVERSARIAL ROUND 3", text)
        self.assertIn("- Add SEC revenue data", text)
        self.assertIn("[NEEDS-RESEARCH]", text)

    def test_merge_without_client_uses_fallback(self):
        text = Synthesizer().merge(
            "t", "PROFILE", ["Add SEC revenue data"], "val", "cha", 1
        )
        self.assertIn("ADVERSARIAL ROUND 1", text)

    def test_llm_failure_falls_back_to_append(self):
        synth = Synthesizer()
        text = synth._merge_with_llm(
            "t", "PROFILE", ["Add SEC revenue data"], "val", "cha", 1, FakeFailingLLM()
        )
        self.assertIn("ADVERSARIAL ROUND 1", text)
        self.assertIn("Add SEC revenue data", text)

    def test_llm_success_returns_merged_content(self):
        lll = ScriptedLLM([success_response("MERGED PROFILE")])
        text = Synthesizer().merge(
            "t", "PROFILE", ["Add SEC revenue data"], "val", "cha", 1, lll
        )
        self.assertEqual(text, "MERGED PROFILE")

    def test_extract_needs_research(self):
        profile = """Some finding here.

[NEEDS-RESEARCH] Verify competitor pricing
[NEEDS-RESEARCH] Interview early adopters
"""
        self.assertEqual(
            Synthesizer().extract_needs_research(profile),
            ["Verify competitor pricing", "Interview early adopters"],
        )

    def test_improvement_delta_counts_confidence_changes(self):
        original = "high confidence stuff\nlow confidence stuff"
        updated = "high confidence stuff\nhigh confidence again"
        delta = Synthesizer().calculate_improvement_delta(original, updated)
        self.assertEqual(delta["high_confidence_change"], 1)
        self.assertEqual(delta["low_confidence_change"], 1)
        self.assertGreater(delta["size_pct"], 0)


class ProfileBuilderTest(unittest.TestCase):
    def test_fallback_marks_degraded_profile(self):
        text = ProfileBuilder()._build_fallback("t", "RAW DATA")
        self.assertIn("No LLM client configured", text)
        self.assertIn("RAW DATA", text)

    def test_llm_failure_falls_back(self):
        profile = ProfileBuilder()
        text = profile._build_with_llm("t", "RAW DATA", FakeFailingLLM())
        self.assertIn("No LLM client configured", text)

    def test_llm_success_returns_profile(self):
        lll = ScriptedLLM([success_response("BUILT PROFILE")])
        text = ProfileBuilder().build("t", "RAW DATA", lll)
        self.assertEqual(text, "BUILT PROFILE")


class ReportGeneratorTest(unittest.TestCase):
    def test_fallback_includes_profile_and_history(self):
        history = [{"round": 1, "improvement_count": 2}]
        text = ReportGenerator()._generate_fallback(
            "Climate policy", "PROFILE BODY", history, 1
        )
        self.assertIn("# Final Research Report: Climate policy", text)
        self.assertIn("went through 1 adversarial rounds", text)
        self.assertIn("PROFILE BODY", text)
        self.assertIn("Round 1: 2 improvements found", text)

    def test_llm_failure_falls_back(self):
        reporter = ReportGenerator()
        text = reporter._generate_with_llm(
            "t", "p", [], "v", "c", 1, FakeFailingLLM()
        )
        self.assertIn("Final Research Report", text)

    def test_llm_success_returns_report(self):
        lll = ScriptedLLM([success_response("FINAL REPORT")])
        text = ReportGenerator().generate("t", "p", [], "v", "c", 1, lll)
        self.assertEqual(text, "FINAL REPORT")

    def test_extract_findings_summary(self):
        report = """# Final Research Report: X

## Key Findings

### Finding 1: Market grew
- **Evidence strength:** Strong

### Finding 2: Adoption rising
- **Evidence strength:** Moderate

### Finding 3: Costs falling
- **Evidence strength:** Weak

## Data & Statistics
"""
        summary = ReportGenerator().extract_findings_summary(report)
        self.assertEqual(summary["total_findings"], 3)
        self.assertEqual(summary["strong"], 1)
        self.assertEqual(summary["moderate"], 1)
        self.assertEqual(summary["weak"], 1)
        self.assertEqual(summary["findings"][0]["title"], "Market grew")


class ScriptedLLM:
    """Tiny chat double: returns queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, **kwargs):
        return self.responses.pop(0)


class FakeFailingLLM:
    """Chat double whose calls always fail."""

    def chat(self, **kwargs):
        return failure_response("boom")


if __name__ == "__main__":
    unittest.main()
