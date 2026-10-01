"""Tests for Team B's improvement-point extraction and vulnerability parsing.

Challenge output is parsed by regex into improvement points that then pass
through the noise filter; a parsing miss silently skips real flaws, and the
vulnerability label feeds the loop's stop decision in the orchestrator.
"""

import unittest

from agent.challenger_team import ChallengerTeam


class ExtractRawPointsTest(unittest.TestCase):
    def setUp(self):
        self.challenger = ChallengerTeam()

    def test_extracts_improve_tagged_points(self):
        text = """## IMPROVEMENT POINTS
[IMPROVE-1] Add Q3 2024 revenue data from SEC filings for Company X to support the growth claim
[IMPROVE-2] Cross-reference the market figure with Gartner or IDC reports
[IMPROVE-3] Remove the claim that demand is universal

## CHALLENGE VERDICT
The growth claim is the biggest vulnerability.
"""
        points = self.challenger._extract_raw_points(text)
        self.assertEqual(len(points), 3)
        self.assertIn("Add Q3 2024 revenue data", points[0])
        self.assertIn("Cross-reference the market figure", points[1])
        self.assertIn("Remove the claim that demand is universal", points[2])

    def test_fallback_parses_numbered_and_bulleted_list(self):
        text = """## IMPROVEMENT POINTS
1. Add Q3 2024 revenue data from SEC filings for Company X
2. - Cross-reference with Gartner reports
* Remove the universal demand claim

## CHALLENGE VERDICT
"""
        points = self.challenger._extract_raw_points(text)
        self.assertEqual(len(points), 3)
        self.assertTrue(points[0].startswith("Add Q3 2024 revenue data"))
        self.assertTrue(points[1].startswith("Cross-reference"))
        self.assertTrue(points[2].startswith("Remove the universal demand claim"))

    def test_text_without_improvements_yields_no_points(self):
        self.assertEqual(
            self.challenger._extract_raw_points("Rate: BULLETPROOF"), []
        )


class ExtractImprovementPointsTest(unittest.TestCase):
    def setUp(self):
        self.challenger = ChallengerTeam()

    def test_noise_is_filtered_out_of_extracted_points(self):
        text = """## IMPROVEMENT POINTS
[IMPROVE-1] Add Q3 2024 revenue data from SEC filings for Company X to support the growth claim
[IMPROVE-2] More research is needed to confirm these findings
"""
        points = self.challenger.extract_improvement_points(text)
        self.assertEqual(len(points), 1)
        self.assertIn("Add Q3 2024 revenue data", points[0])

    def test_fallback_challenge_yields_no_actionable_points(self):
        # The no-LLM fallback's own improvement point is not actionable, so
        # degraded-mode runs auto-validate after a single round.
        points = self.challenger.extract_improvement_points(
            self.challenger._challenge_fallback("topic", "profile", 1)
        )
        self.assertEqual(points, [])

    def test_unfiltered_extraction_returns_raw_points(self):
        text = "[IMPROVE-1] Add Q3 2024 revenue data from SEC filings"
        raw = self.challenger.extract_improvement_points(text, filter_noise=False)
        self.assertEqual(raw, ["Add Q3 2024 revenue data from SEC filings"])


class AssessVulnerabilityTest(unittest.TestCase):
    def setUp(self):
        self.challenger = ChallengerTeam()

    def test_classifies_vulnerability_levels(self):
        cases = [
            ("Rate: BULLETPROOF", "bulletproof"),
            ("Rate: CRITICALLY WEAK", "critically_weak"),
            ("Rate: SIGNIFICANT FLAWS", "significant_flaws"),
            ("Rate: MINOR GAPS", "minor_gaps"),
        ]
        for text, expected in cases:
            self.assertEqual(self.challenger.assess_vulnerability(text), expected)

    def test_unknown_when_no_rating_present(self):
        self.assertEqual(
            self.challenger.assess_vulnerability("no rating here"), "unknown"
        )


class FallbackChallengeTest(unittest.TestCase):
    def test_fallback_marks_unknown_and_asks_for_key(self):
        text = ChallengerTeam()._challenge_fallback("t", "p", 2)
        self.assertIn("Rate: UNKNOWN", text)
        self.assertIn("[IMPROVE-1]", text)
        self.assertIn("Round: 2", text)


if __name__ == "__main__":
    unittest.main()
