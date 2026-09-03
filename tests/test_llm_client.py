"""Tests for LLM client retry/cost behavior and client factory logic.

The LLM client is the money and reliability layer of every pipeline run:
token usage drives cost accounting, and the retry loop decides whether an
API failure surfaces as a graceful degraded result or an exception.
"""

import os
import unittest
from unittest import mock

from agent.llm_client import (
    LLMClient,
    calculate_cost,
    create_client_from_config,
    estimate_tokens,
)

from tests.helpers import FakeResponse


class CalculateCostTest(unittest.TestCase):
    def test_known_model_uses_its_pricing_table(self):
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        self.assertAlmostEqual(
            calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000), 0.75, places=6
        )

    def test_unknown_model_uses_conservative_default_pricing(self):
        # DEFAULT_PRICING: $3.00/1M input, $10.00/1M output
        self.assertAlmostEqual(
            calculate_cost("some-future-model", 1_000_000, 500_000), 8.0, places=6
        )

    def test_zero_tokens_cost_zero(self):
        self.assertEqual(calculate_cost("gpt-4o-mini", 0, 0), 0.0)

    def test_estimate_tokens_returns_positive_int(self):
        self.assertGreaterEqual(estimate_tokens("hello world"), 1)
        self.assertIsInstance(estimate_tokens("hello world"), int)


class LLMClientChatTest(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(
            api_key="sk-test",
            base_url="https://api.openai.com/v1/",
            model="gpt-4o-mini",
            max_retries=3,
        )
        # Avoid real backoff sleeps in retry-path tests.
        patcher = mock.patch("agent.llm_client.time.sleep")
        self.mock_sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_tracks_tokens_and_cost(self):
        response = FakeResponse(
            status_code=200,
            json_data={
                "choices": [{"message": {"content": "Hello world"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )
        with mock.patch("agent.llm_client.requests.post", return_value=response) as post:
            result = self.client.chat(
                [{"role": "user", "content": "ping"}], system="be brief"
            )

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["content"], "Hello world")
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["output_tokens"], 500)
        # gpt-4o-mini: 1000*0.15/1M + 500*0.60/1M
        self.assertAlmostEqual(result["cost"], 0.00045, places=6)

        # Request payload splits system out of the user messages.
        url, kwargs = post.call_args
        self.assertTrue(url[0].endswith("/chat/completions"))
        sent_messages = kwargs["json"]["messages"]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(sent_messages[0]["content"], "be brief")

        self.assertEqual(len(self.client.call_log), 1)
        self.assertEqual(self.client.call_log[0]["attempt"], 1)
        self.assertEqual(self.client.total_input_tokens, 1000)
        self.assertEqual(self.client.total_output_tokens, 500)
        summary = self.client.get_cost_summary()
        self.assertEqual(summary["total_calls"], 1)
        self.assertEqual(summary["total_tokens"], 1500)

    def test_retries_then_succeeds_and_logs_attempt_number(self):
        def flaky_post(*args, **kwargs):
            flaky_post.calls += 1
            if flaky_post.calls == 1:
                raise ConnectionError("temporary network failure")
            return FakeResponse(
                status_code=200,
                json_data={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        flaky_post.calls = 0
        with mock.patch("agent.llm_client.requests.post", side_effect=flaky_post):
            result = self.client.chat([{"role": "user", "content": "ping"}])

        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "ok")
        self.assertEqual(len(self.client.call_log), 1)
        self.assertEqual(self.client.call_log[0]["attempt"], 2)
        self.mock_sleep.assert_called_once_with(1)

    def test_exhausted_retries_return_graceful_failure_without_cost(self):
        response = FakeResponse(status_code=429, json_data=None, text="rate limited")
        with mock.patch(
            "agent.llm_client.requests.post", return_value=response
        ) as post:
            result = self.client.chat([{"role": "user", "content": "ping"}])

        self.assertFalse(result["success"])
        self.assertIn("429", result["error"])
        self.assertEqual(result["content"], "")
        self.assertEqual(post.call_count, 3)  # max_retries attempts
        self.assertEqual(self.client.total_cost, 0.0)
        self.assertEqual(self.client.call_log, [])

    def test_transport_exception_finally_fails(self):
        with mock.patch(
            "agent.llm_client.requests.post",
            side_effect=ConnectionError("offline"),
        ):
            result = self.client.chat([{"role": "user", "content": "ping"}])

        self.assertFalse(result["success"])
        self.assertIn("offline", result["error"])
        self.assertEqual(self.client.total_cost, 0.0)

    def test_anthropic_endpoint_uses_messages_api(self):
        client = LLMClient(
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet-4",
            max_retries=1,
        )
        response = FakeResponse(
            status_code=200,
            json_data={
                "content": [{"text": "claude answers"}],
                "usage": {"input_tokens": 20, "output_tokens": 7},
            },
        )
        with mock.patch("agent.llm_client.requests.post", return_value=response) as post:
            result = client.chat(
                [{"role": "user", "content": "hi"}], system="be terse"
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "claude answers")
        url, kwargs = post.call_args
        self.assertTrue(url[0].endswith("/messages"))
        self.assertEqual(kwargs["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(kwargs["json"]["system"], "be terse")
        self.assertEqual(len(kwargs["json"]["messages"]), 1)  # system removed
        self.assertEqual(result["input_tokens"], 20)


class CreateClientFromConfigTest(unittest.TestCase):
    def setUp(self):
        # A clean environment has these absent; "" would defeat the defaults.
        self._saved_env = {
            k: os.environ.get(k)
            for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME")
        }
        for k in self._saved_env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_config_values_win_over_environment(self):
        client = create_client_from_config(
            {
                "api_key": "config-key",
                "base_url": "https://config.example.com/v1",
                "model_name": "gpt-4o",
            }
        )
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "config-key")
        self.assertEqual(client.base_url, "https://config.example.com/v1")
        self.assertEqual(client.model, "gpt-4o")

    def test_environment_fallback(self):
        with mock.patch.dict(
            os.environ,
            {"LLM_API_KEY": "env-key", "LLM_BASE_URL": "https://env.example.com"},
            clear=False,
        ):
            client = create_client_from_config({})
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "env-key")
        self.assertEqual(client.base_url, "https://env.example.com")

    def test_empty_config_defaults_base_url_and_model(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "key-only"}, clear=False):
            client = create_client_from_config({})
        self.assertIsNotNone(client)
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.model, "gpt-4o-mini")

    def test_missing_api_key_returns_none(self):
        self.assertIsNone(create_client_from_config({"base_url": "https://x.example.com"}))

    def test_trailing_slash_is_stripped(self):
        client = create_client_from_config(
            {"api_key": "k", "base_url": "https://x.example.com/v1/"}
        )
        self.assertEqual(client.base_url, "https://x.example.com/v1")


if __name__ == "__main__":
    unittest.main()
