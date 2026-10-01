"""Tests for the orchestration control flow and DB persistence.

run() wires the five phases together, decides when the adversarial loop
stops, and persists every step to SQLite. These tests swap phase
collaborators for scripted doubles and point the database at a temp file so
the real loop logic is exercised end to end without network or API calls.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from agent import orchestrator
from agent.orchestrator import ResearchPipeline

from tests.helpers import FakeLLMClient, success_response

ACTIONABLE_POINT = (
    "Add Q3 2024 revenue data from SEC filings to support the growth claim"
)
CHALLENGE_WITH_FLAW = f"[IMPROVE-1] {ACTIONABLE_POINT}"
CHALLENGE_OK = "## CHALLENGE SUMMARY\nRate: BULLETPROOF — no flaws found"


class FakeResearch:
    def execute(self, topic, num_agents=4, run_id="", log_fn=None, llm_client=None):
        return f"# Research Data: {topic}\nhttps://source.example.org/facts"


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(
            orchestrator, "DB_PATH", os.path.join(self._tmp.name, "research.db")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        # Isolate from any developer machine env keys / .env content.
        keys = [
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL_NAME",
            "MAX_ADVERSARIAL_ROUNDS",
            "MAX_RESEARCH_AGENTS",
        ]
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)

    def _pipeline(self, config=None):
        pipeline = ResearchPipeline(
            topic="Email API market", config=config or {"max_rounds": 3}
        )
        pipeline.research = FakeResearch()
        return pipeline

    def _rows(self, table):
        conn = orchestrator.get_db()
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        finally:
            conn.close()


class FullRunTest(PipelineTestCase):
    def test_loop_breaks_when_challenger_finds_no_flaws(self):
        client = FakeLLMClient(
            responses=[
                success_response("BUILT PROFILE"),
                success_response("VALIDATION ROUND 1"),
                success_response(CHALLENGE_WITH_FLAW),
                success_response("MERGED ROUND 1 PROFILE"),
                success_response("VALIDATION ROUND 2"),
                success_response(CHALLENGE_OK),
                success_response("# Final Research Report"),
            ]
        )
        pipeline = self._pipeline()
        pipeline.llm_client = client

        result = pipeline.run()

        self.assertTrue(result["success"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["final_report"], "# Final Research Report")
        self.assertEqual(result["profile"], "MERGED ROUND 1 PROFILE")

        # Round 1 found one flaw, round 2 validated cleanly, loop stopped.
        self.assertEqual(result["history"][0]["improvement_points"], [ACTIONABLE_POINT])
        self.assertEqual(result["history"][0]["improvement_count"], 1)
        self.assertEqual(result["history"][1]["improvement_count"], 0)

        # 7 LLM calls: build, val, cha, merge, val, cha, report.
        self.assertEqual(len(client.chat_calls), 7)
        # Round 2 validator received the merged profile, proving the loop wires
        # synthesis output back into the next interrogation.
        self.assertIn(
            "MERGED ROUND 1 PROFILE", client.chat_calls[4]["messages"][0]["content"]
        )
        self.assertEqual(result["cost"]["total_calls"], 7)

    def test_every_phase_is_persisted(self):
        client = FakeLLMClient(
            responses=[
                success_response("BUILT PROFILE"),
                success_response("VALIDATION"),
                success_response(CHALLENGE_OK),
                success_response("# Final Research Report"),
            ]
        )
        pipeline = self._pipeline()
        pipeline.llm_client = client

        result = pipeline.run()
        self.assertTrue(result["success"])

        run_row = self._rows("research_runs")[0]
        self.assertEqual(run_row["status"], "completed")
        self.assertEqual(run_row["current_phase"], "completed")
        self.assertIsNotNone(run_row["final_report"])

        phases = self._rows("phase_results")
        self.assertEqual(
            [p["phase"] for p in phases],
            ["research", "profile", "adversarial", "adversarial"],
        )

        subagents = self._rows("subagent_logs")
        self.assertEqual(len(subagents), 2)
        for row in subagents:
            self.assertEqual(row["status"], "completed")
            self.assertIsNotNone(row["completed_at"])

        cost_rows = self._rows("cost_logs")
        self.assertEqual(
            [c["phase"] for c in cost_rows],
            [
                "research",
                "profile",
                "adversarial_r1_validator",
                "adversarial_r1_challenger",
                "report",
            ],
        )


class DegradedRunTest(PipelineTestCase):
    def test_degraded_mode_completes_in_single_round(self):
        pipeline = self._pipeline()
        self.assertIsNone(pipeline.llm_client)

        result = pipeline.run()

        self.assertTrue(result["success"])
        # The no-LLM fallback's own improvement point is not actionable, so the
        # loop validates after one round.
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["history"][0]["improvement_count"], 0)
        self.assertEqual(result["cost"], {})
        self.assertIn("Final Research Report", result["final_report"])
        self.assertIn("No LLM client configured", result["profile"])
        self.assertEqual(self._rows("research_runs")[0]["status"], "completed")


class ErrorHandlingTest(PipelineTestCase):
    def test_phase_failure_marks_run_as_error(self):
        def explode(**kwargs):
            raise RuntimeError("LLM provider down")

        pipeline = self._pipeline()
        pipeline.research = FakeResearch()
        pipeline.validator.validate = explode

        result = pipeline.run()

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "LLM provider down")
        self.assertEqual(result["run_id"], pipeline.run_id)
        self.assertEqual(self._rows("research_runs")[0]["status"], "error")

    def test_get_status_returns_not_found_for_unknown_run(self):
        status = self._pipeline().get_status()
        self.assertEqual(status["status"], "not_found")


class MaxRoundsTest(PipelineTestCase):
    def test_loop_stops_at_max_rounds_when_flaws_never_end(self):
        client = FakeLLMClient(
            responses=[
                success_response("BUILT PROFILE"),
                success_response("VALIDATION R1"),
                success_response(CHALLENGE_WITH_FLAW),
                success_response("MERGED R1"),
                success_response("VALIDATION R2"),
                success_response(CHALLENGE_WITH_FLAW),
                success_response("MERGED R2"),
                success_response("# Final Research Report"),
            ]
        )
        pipeline = self._pipeline(config={"max_rounds": 2})
        pipeline.llm_client = client

        result = pipeline.run()

        self.assertTrue(result["success"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual([h["round"] for h in result["history"]], [1, 2])
        self.assertEqual(result["history"][1]["improvement_count"], 1)
        self.assertEqual(result["profile"], "MERGED R2")
        # build + 2*(validate+challenge+merge) + report
        self.assertEqual(len(client.chat_calls), 8)
        self.assertEqual(self._rows("research_runs")[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
