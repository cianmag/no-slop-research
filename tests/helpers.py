"""Shared test doubles for the agent pipeline."""


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class FakeLLMClient:
    """Scripted LLM client: returns queued responses, records chat calls."""

    def __init__(self, responses=None, model="gpt-4o-mini"):
        self.responses = list(responses) if responses else []
        self.chat_calls = []
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=4096, system=None):
        self.chat_calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
        })
        if self.responses:
            return self.responses.pop(0)
        return {
            "content": "fake content",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost": 0.0,
            "model": self.model,
            "duration_ms": 1,
            "success": True,
            "error": None,
        }

    def get_cost_summary(self):
        total_calls = len(self.chat_calls)
        return {
            "model": self.model,
            "total_calls": total_calls,
            "total_input_tokens": 10 * total_calls,
            "total_output_tokens": 5 * total_calls,
            "total_tokens": 15 * total_calls,
            "total_cost_usd": 0.0,
            "avg_cost_per_call": 0.0,
            "calls": [],
        }


def success_response(content, input_tokens=100, output_tokens=50):
    """Build a chat result dict shaped like LLMClient.chat's success payload."""
    return {
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": 0.0,
        "model": "gpt-4o-mini",
        "duration_ms": 1,
        "success": True,
        "error": None,
    }


def failure_response(error="boom"):
    """Build a chat result dict shaped like LLMClient.chat's failure payload."""
    return {
        "content": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "model": "gpt-4o-mini",
        "duration_ms": 1,
        "success": False,
        "error": error,
    }
