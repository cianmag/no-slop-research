"""Tests for the evidence-based confidence scorer.

The confidence label (high/moderate/low) is the headline users see on every
run, so the scoring components and thresholds are high-regression-risk.
"""

import unittest

from agent.confidence_scorer import (
    _extract_claim_scores,
    _score_corroboration,
    _score_source_quality,
    _score_weaknesses,
    score_confidence,
)


class ScoreComponentsTest(unittest.TestCase):
    def test_source_quality_counts_unique_domains(self):
        text = " ".join(f"https://source{i}.example.org/report" for i in range(10))
        self.assertEqual(_score_source_quality(text, ""), 1.0)

    def test_source_quality_zero_without_sources(self):
        self.assertEqual(_score_source_quality("no urls here", "profile text"), 0.0)

    def test_corroboration_reduced_by_contradictions(self):
        agreeing = "Several sources agree with these findings"
        contradiction = " Other findings contradict the central claim."
        self.assertGreater(
            _score_corroboration(agreeing), _score_corroboration(agreeing + contradiction)
        )

    def test_weakness_penalty_is_capped_at_one(self):
        challenge = "critical " * 20  # 20 * 0.20 far exceeds the 1.0 cap
        self.assertEqual(_score_weaknesses(challenge), 1.0)

    def test_weakness_penalty_zero_without_challenge(self):
        self.assertEqual(_score_weaknesses(""), 0.0)

    def test_bulletproof_reduces_penalty(self):
        self.assertEqual(_score_weaknesses("The research is bulletproof"), 0.0)


class ScoreConfidenceTest(unittest.TestCase):
    def test_rich_evidence_scores_high(self):
        validation = """
Peer-reviewed journal studies confirm these results consistently.
Multiple sources agree on the market size.
Strong evidence supports the central findings.
https://journal.example.org/study
https://gov.example.org/data
https://university.example.org/research
https://institute.example.org/report
https://review.example.org/analysis
https://market.example.org/stats
"""
        result = score_confidence(validation, validation)
        self.assertEqual(result["confidence_label"], "high")
        self.assertGreaterEqual(result["overall_score"], 0.75)

    def test_sparse_evidence_scores_moderate(self):
        validation = """Several studies agree with these numbers.
The findings are consistent across reports.
Moderate evidence supports the core claim.
https://alpha.example.org/1
https://beta.example.org/1
https://gamma.example.org/1
"""
        result = score_confidence(validation, validation)
        self.assertEqual(result["confidence_label"], "moderate")
        self.assertGreaterEqual(result["overall_score"], 0.50)
        self.assertLess(result["overall_score"], 0.75)

    def test_empty_input_scores_low(self):
        result = score_confidence("", "")
        self.assertEqual(result["confidence_label"], "low")
        self.assertLess(result["overall_score"], 0.50)

    def test_critical_challenge_drags_score_to_low(self):
        validation = "https://a.example.org https://b.example.org"
        challenge = "critically weak research with fatal severe fundamental flaws"
        result = score_confidence(validation, validation, challenge)
        self.assertEqual(result["confidence_label"], "low")
        self.assertEqual(result["weakness_penalty"], 1.0)

    def test_overall_score_stays_within_zero_one(self):
        challenge = "critical severe fatal fundamental " * 10
        result = score_confidence(
            "https://a.example.org " * 30,
            "multiple sources agree that strong evidence exists",
            challenge,
        )
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)

    def test_methodology_describes_weights(self):
        result = score_confidence("", "")
        self.assertIn("30%", result["methodology"])
        self.assertIn("25%", result["methodology"])


class ClaimScoresTest(unittest.TestCase):
    def test_extracts_claim_scores_with_evidence(self):
        validation = """**Claim:** The market grew 12% last year.
**Evidence Quality:** Strong
**Confidence Rating:** 85%

**Claim:** AI adoption is accelerating
Evidence Quality: Moderate
Confidence Rating: High

**Claim:** Costs are falling everywhere
**Evidence Quality:** Weak
**Confidence Rating:** Low
"""
        claims = _extract_claim_scores(validation)
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0]["claim"], "The market grew 12% last year.")
        self.assertEqual(claims[0]["score"], 0.85)
        self.assertEqual(claims[0]["evidence"], "strong")
        self.assertEqual(claims[1]["score"], 0.8)
        self.assertEqual(claims[1]["evidence"], "moderate")
        self.assertEqual(claims[2]["score"], 0.3)
        self.assertEqual(claims[2]["evidence"], "weak")

    def test_claim_without_evidence_line_is_unspecified(self):
        validation = """**Claim:** Something happened.
**Confidence Rating:** 70%
"""
        claims = _extract_claim_scores(validation)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["evidence"], "unspecified")
        self.assertEqual(claims[0]["score"], 0.7)

    def test_no_claims_returns_empty_list(self):
        self.assertEqual(_extract_claim_scores("no claim blocks here"), [])


if __name__ == "__main__":
    unittest.main()
