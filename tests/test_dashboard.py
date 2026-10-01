"""Tests for the Flask dashboard's REST endpoints.

The dashboard is how users launch pipelines and manage API keys, yet none of
its route handlers had coverage. Tests exercise the handlers through the
Flask test client with a temp database and a stubbed pipeline class, so no
network or LLM calls happen.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from dashboard import app as dashboard_app
from agent import orchestrator


class FakePipeline:
    last_config = None

    def __init__(self, topic, config=None, run_id=None):
        self.topic = topic
        self.run_id = "testrun01"
        FakePipeline.last_config = config

    def run(self):
        return {
            "success": True,
            "run_id": self.run_id,
            "rounds": 1,
            "final_report": "short report",
            "history": [],
            "profile": "profile",
            "confidence": {"confidence_label": "low", "overall_score": 0.3},
            "cost": {},
        }


class FailedPipeline(FakePipeline):
    def run(self):
        return {"success": False, "error": "provider exploded"}


class InlineThread:
    """Run background targets inline so queue updates are deterministic."""

    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


class DashboardTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = os.path.join(self._tmp.name, "research.db")
        # init_db routes through orchestrator.get_db, while route handlers use
        # the dashboard's imported DB_PATH — keep both pointing at the temp db.
        for module in (dashboard_app, orchestrator):
            patcher = mock.patch.object(module, "DB_PATH", db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

        for key in (
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL_NAME",
            "MAX_ADVERSARIAL_ROUNDS",
        ):
            self.addCleanup(os.environ.pop, key, None)
            os.environ.pop(key, None)

        dashboard_app.init_db()
        self.client = dashboard_app.app.test_client()

    def _seed_run(self, run_id="abc12345", report="report body"):
        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            now = "2026-01-01T00:00:00+00:00"
            conn.execute(
                "INSERT INTO research_runs (id, topic, status, current_phase, max_rounds, created_at, updated_at, final_report, config) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, "topic", "completed", "completed", 3, now, now, report, "{}"),
            )
            conn.commit()
        finally:
            conn.close()


class KeyManagementTest(DashboardTestCase):
    def test_add_key_requires_value(self):
        rv = self.client.post("/api/keys", json={"provider": "openai"})
        self.assertEqual(rv.status_code, 400)
        self.assertIn("key_value", rv.get_json()["error"])

    def test_add_key_autofills_base_url_from_preset(self):
        rv = self.client.post(
            "/api/keys",
            json={"provider": "openai", "key_value": "sk-test-1234567890"},
        )
        self.assertEqual(rv.status_code, 200)
        keys = self.client.get("/api/keys").get_json()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0]["provider"], "openai")
        self.assertEqual(keys[0]["base_url"], "https://api.openai.com/v1")
        # Raw value is never exposed; masked form only.
        self.assertEqual(keys[0]["key_masked"], "sk-t...7890")
        self.assertNotIn("sk-test-1234567890", json.dumps(keys))

    def test_toggle_and_delete_key(self):
        self.client.post("/api/keys", json={"provider": "groq", "key_value": "gsk_x"})
        key_id = self.client.get("/api/keys").get_json()[0]["id"]

        self.client.post(f"/api/keys/{key_id}/toggle")
        self.assertFalse(self.client.get("/api/keys").get_json()[0]["is_active"])

        self.client.delete(f"/api/keys/{key_id}")
        self.assertEqual(self.client.get("/api/keys").get_json(), [])

    def test_providers_presets_returned(self):
        rv = self.client.get("/api/providers")
        self.assertEqual(rv.status_code, 200)
        self.assertIn("openai", rv.get_json())
        self.assertIn("custom", rv.get_json())


class ResearchEndpointsTest(DashboardTestCase):
    def test_start_research_rejects_empty_topic(self):
        rv = self.client.post("/api/research/start", json={"topic": "   "})
        self.assertEqual(rv.status_code, 400)
        self.assertEqual(rv.get_json()["error"], "topic is required")

    def test_start_research_runs_pipeline_and_queues_it(self):
        with mock.patch.object(dashboard_app, "ResearchPipeline", FakePipeline), \
             mock.patch.object(dashboard_app, "create_client_from_env", return_value=object()), \
             mock.patch.object(dashboard_app.threading, "Thread", InlineThread):
            rv = self.client.post(
                "/api/research/start",
                json={"topic": "Email API market", "config": {"max_rounds": 2}},
            )

        self.assertEqual(rv.status_code, 200)
        self.assertTrue(rv.get_json()["success"])
        self.assertEqual(rv.get_json()["run_id"], "testrun01")
        self.assertEqual(FakePipeline.last_config, {"max_rounds": 2})

        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            row = conn.execute(
                "SELECT status FROM research_queue WHERE run_id = ?", ("testrun01",)
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row[0], "completed")

    def test_start_research_injects_active_key_into_config(self):
        self.client.post(
            "/api/keys", json={"provider": "openai", "key_value": "sk-active-key"}
        )
        with mock.patch.object(dashboard_app, "ResearchPipeline", FakePipeline), \
             mock.patch.object(dashboard_app.threading, "Thread", InlineThread):
            self.client.post(
                "/api/research/start", json={"topic": "Market report"}
            )
        self.assertEqual(FakePipeline.last_config["api_key"], "sk-active-key")
        self.assertEqual(FakePipeline.last_config["provider"], "openai")

    def test_start_research_pipeline_error_marks_queue_error(self):
        with mock.patch.object(dashboard_app, "ResearchPipeline", FailedPipeline), \
             mock.patch.object(dashboard_app, "create_client_from_env", return_value=object()), \
             mock.patch.object(dashboard_app.threading, "Thread", InlineThread):
            rv = self.client.post(
                "/api/research/start", json={"topic": "Doomed topic"}
            )

        self.assertEqual(rv.status_code, 200)
        self.assertTrue(rv.get_json()["success"])

        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            row = conn.execute(
                "SELECT status FROM research_queue WHERE run_id = ?", ("testrun01",)
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row[0], "error")

    def test_get_run_returns_404_for_unknown_run(self):
        rv = self.client.get("/api/research/runs/nope")
        self.assertEqual(rv.status_code, 404)


class RunDetailTest(DashboardTestCase):
    def test_get_run_sanitizes_control_characters_and_truncates(self):
        self._seed_run(report="clean \x00body\r\nwith control chars")
        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            conn.execute(
                "INSERT INTO phase_results (run_id, phase, round_num, team, result, created_at) VALUES (?, 'adversarial', 1, 'validator', ?, 'now')",
                ("abc12345", "x" * 2500 + "\x00\x01END"),
            )
            conn.commit()
        finally:
            conn.close()

        rv = self.client.get("/api/research/runs/abc12345")
        self.assertEqual(rv.status_code, 200)
        text = rv.get_data(as_text=True)

        self.assertNotIn("\x00", text)
        self.assertNotIn("\x01", text)
        self.assertNotIn("\r", text)
        self.assertIn("with control chars", text)
        self.assertIn("[... truncated", text)

    def test_delete_run_cascades_to_child_tables(self):
        self._seed_run()
        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            conn.execute(
                "INSERT INTO phase_results (run_id, phase, round_num, team, result, created_at) VALUES ('abc12345', 'research', 0, 'research', 'r', 'now')"
            )
            conn.execute(
                "INSERT INTO subagent_logs (run_id, phase, round_num, agent_role, agent_index, status, started_at) VALUES ('abc12345', 'adversarial', 1, 'validator', 0, 'completed', 'now')"
            )
            conn.execute(
                "INSERT INTO research_queue (topic, config, status, run_id, created_at) VALUES ('t', '{}', 'completed', 'abc12345', 'now')"
            )
            conn.commit()
        finally:
            conn.close()

        rv = self.client.delete("/api/research/runs/abc12345")
        self.assertEqual(rv.status_code, 200)

        conn = sqlite3.connect(dashboard_app.DB_PATH)
        try:
            counts = []
            for table, key in (
                ("research_runs", "id"),
                ("phase_results", "run_id"),
                ("subagent_logs", "run_id"),
                ("research_queue", "run_id"),
            ):
                counts.append(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {key} = 'abc12345'"
                    ).fetchone()[0]
                )
        finally:
            conn.close()
        self.assertEqual(counts, [0, 0, 0, 0])


class SettingsAndHealthTest(DashboardTestCase):
    def test_settings_roundtrip(self):
        self.assertEqual(self.client.get("/api/settings").get_json(), {})
        rv = self.client.post("/api/settings", json={"max_rounds": "5"})
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(
            self.client.get("/api/settings").get_json(), {"max_rounds": "5"}
        )

    def test_health_reports_db_and_status(self):
        rv = self.client.get("/api/health")
        self.assertEqual(rv.status_code, 200)
        body = rv.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["db"])

    def test_index_renders_dashboard(self):
        rv = self.client.get("/")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"No-Slop Research Dashboard", rv.data)
        self.assertIn(b"Start New Research", rv.data)

    def test_mask_key_preserves_ends_of_long_keys(self):
        self.assertEqual(dashboard_app.mask_key(""), "***")
        self.assertEqual(dashboard_app.mask_key("short"), "***")
        self.assertEqual(
            dashboard_app.mask_key("sk-test-1234567890"), "sk-t...7890"
        )


if __name__ == "__main__":
    unittest.main()
