"""Tests for the noise filter that gates Team B improvement points.

filter_improvement_points decides what counts as an actionable flaw; the
orchestrator breaks the adversarial loop when the filtered list is empty,
so false-positive filtering silently converts challenged research into
"validated" output.
"""

import unittest

from agent.noise_filter import (
    _is_actionable,
    _is_generic_noise,
    _similarity,
    filter_improvement_points,
    severity_score,
)


class FilterImprovementPointsTest(unittest.TestCase):
    def test_empty_points_return_empty_result(self):
        result = filter_improvement_points([], topic="anything")
        self.assertEqual(result["filtered_points"], [])
        self.assertEqual(result["removed_noise"], [])
        self.assertEqual(result["quality_score"], 0.0)

    def test_keeps_specific_actionable_points(self):
        points = [
            "Add Q3 2024 revenue data from SEC filings for Company X to support the growth claim",
            "Cross-reference the market size figure with Gartner or IDC reports",
        ]
        result = filter_improvement_points(points, topic="email API market")
        self.assertEqual(result["filtered_points"], points)
        self.assertEqual(result["removed_noise"], [])
        self.assertEqual(result["quality_score"], 1.0)

    def test_removes_too_short_points(self):
        result = filter_improvement_points(["Add sources."], topic="x")
        self.assertEqual(result["filtered_points"], [])
        self.assertEqual(result["removed_noise"], ["Add sources."])
        # The rejection reason must name the actual threshold, not a literal
        # "{min_length}" placeholder (regression: the string was not an f-string).
        self.assertIn("Too short (< 20 chars)", result["removed_reasons"]["Add sources."])

    def test_removes_generic_noise(self):
        points = [
            "More research is needed before drawing conclusions",
            "It would be helpful to include more sources",
        ]
        result = filter_improvement_points(points, topic="x")
        self.assertEqual(result["filtered_points"], [])
        for point in points:
            self.assertEqual(
                result["removed_reasons"][point[:50]],
                "Generic noise — no actionable content",
            )

    def test_removes_non_actionable_points(self):
        point = "We should add a chart to illustrate this."
        # "add" is the only actionable signal — one match is not enough.
        self.assertFalse(_is_actionable(point))
        result = filter_improvement_points([point], topic="x")
        self.assertEqual(result["filtered_points"], [])
        self.assertEqual(
            result["removed_reasons"][point[:50]],
            "Not actionable — no specific fix described",
        )

    def test_deduplicates_near_duplicate_points(self):
        first = "Add Q3 2024 revenue data from SEC filings to support the growth claim"
        second = "Add Q3 2024 revenue data from SEC filings to support the growth claims"
        self.assertGreater(_similarity(first, second), 0.7)

        result = filter_improvement_points([first, second], topic="x")
        self.assertEqual(result["filtered_points"], [first])
        self.assertEqual(result["removed_noise"], [second])
        self.assertIn("Duplicate of", result["removed_reasons"][second[:50]])

    def test_quality_score_is_kept_ratio(self):
        points = [
            "Add Q3 2024 revenue data from SEC filings to support the growth claim",
            "More research is needed before drawing conclusions",
        ]
        result = filter_improvement_points(points, topic="x")
        self.assertEqual(result["quality_score"], 0.5)

    def test_filler_heavy_points_are_noise(self):
        point = " ".join(["the", "and", "more", "of", "in", "to", "would", "be"] * 3)
        self.assertGreater(len(point.split()), 15)
        self.assertTrue(_is_generic_noise(point))
        result = filter_improvement_points([point], topic="x")
        self.assertEqual(result["filtered_points"], [])


class ActionabilityTest(unittest.TestCase):
    def test_requires_two_signal_matches(self):
        # Two independent signals: "add" + "data".
        self.assertTrue(
            _is_actionable("Add 2024 revenue data to support the growth claim")
        )
        self.assertFalse(_is_actionable("We should add a chart."))

    def test_improv_tag_alone_is_not_enough(self):
        # The [IMPROVE-N] tag is one signal, but actionability needs two.
        self.assertFalse(_is_actionable("[IMPROVE-1] do the thing carefully now"))

    def test_improv_tag_with_substance_passes(self):
        self.assertTrue(
            _is_actionable("[IMPROVE-1] Add missing revenue data from SEC filings")
        )


class SeverityScoreTest(unittest.TestCase):
    def test_critical_missing_evidence(self):
        self.assertEqual(
            severity_score("Missing source citations for the growth claim"), "critical"
        )

    def test_critical_false_or_misleading(self):
        self.assertEqual(severity_score("The framing is misleading and incorrect"), "critical")

    def test_major_outdated_data(self):
        self.assertEqual(severity_score("The analysis relies on outdated figures"), "major")

    def test_major_biased_framing(self):
        self.assertEqual(severity_score("The conclusion is biased toward one vendor"), "major")

    def test_minor_for_unremarkable_points(self):
        self.assertEqual(severity_score("The report could use a clearer title"), "minor")


if __name__ == "__main__":
    unittest.main()
