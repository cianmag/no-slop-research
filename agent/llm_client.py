"""
No-Slop Research — LLM Client

Direct API client for OpenAI-compatible endpoints (OpenAI, OpenRouter,
Groq, Together, DeepSeek, custom). Handles retries, timeouts, error
reporting, and token counting.

This is the core engine that makes agents actually work standalone.
"""

import json
import time
import os
import re
from typing import Optional

import requests

# Token pricing per 1M tokens (input/output) — updated May 2025
PRICING = {
    # OpenAI
    "gpt-4o":               {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":          {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":        {"input": 0.50,  "output": 1.50},
    # Anthropic
    "claude-sonnet-4":      {"input": 3.00,  "output": 15.00},
    "claude-3.5-sonnet":    {"input": 3.00,  "output": 15.00},
    "claude-3.5-haiku":     {"input": 0.80,  "output": 4.00},
    "claude-3-opus":        {"input": 15.00, "output": 75.00},
    # Google
    "gemini-2.5-pro":       {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":     {"input": 0.15,  "output": 0.60},
    "gemini-pro-1.5":       {"input": 1.25,  "output": 5.00},
    # Meta / Groq / Together
    "llama-3.1-70b-versatile":  {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":       {"input": 0.24, "output": 0.24},
    # DeepSeek
    "deepseek-chat":        {"input": 0.14, "output": 0.28},
    "deepseek-coder":       {"input": 0.14, "output": 0.28},
    "deepseek-reasoner":    {"input": 0.55, "output": 2.19},
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = {"input": 3.00, "output": 10.00}


def estimate_tokens(text: str) -> int:
    """Rough token count (1 token ≈ 4 chars for English)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


class LLMClient:
    """
    Direct LLM API client. Supports any OpenAI-compatible endpoint.
    No dependencies on Hermes or external agent frameworks.
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_retries: int = 3, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_log = []

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 4096, system: str = None) -> dict:
        """
        Send a chat completion request.

        Returns:
            {
                "content": str,
                "input_tokens": int,
                "output_tokens": int,
                "cost": float,
                "model": str,
                "duration_ms": int,
                "success": bool,
                "error": str | None
            }
        """
        start_time = time.time()

        # Build message list
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Determine if this is an Anthropic endpoint
        is_anthropic = "anthropic.com" in self.base_url

        for attempt in range(self.max_retries):
            try:
                if is_anthropic:
                    result = self._call_anthropic(full_messages, temperature, max_tokens)
                else:
                    result = self._call_openai_compatible(full_messages, temperature, max_tokens)

                duration_ms = int((time.time() - start_time) * 1000)

                # Track costs
                self.total_input_tokens += result.get("input_tokens", 0)
                self.total_output_tokens += result.get("output_tokens", 0)
                cost = calculate_cost(self.model, result.get("input_tokens", 0),
                                      result.get("output_tokens", 0))
                self.total_cost += cost

                call_record = {
                    "model": self.model,
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "cost": cost,
                    "duration_ms": duration_ms,
                    "attempt": attempt + 1
                }
                self.call_log.append(call_record)

                return {
                    "content": result.get("content", ""),
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "cost": cost,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue

                duration_ms = int((time.time() - start_time) * 1000)
                return {
                    "content": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "success": False,
                    "error": str(e)
                }

    def _call_openai_compatible(self, messages: list, temperature: float,
                                 max_tokens: int) -> dict:
        """Call any OpenAI-compatible API using requests (avoids httpx decompression issues)."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity"  # Avoid decompression issues
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        usage = data.get("usage", {})
        choice = data["choices"][0]
        message = choice.get("message", {})

        # Handle reasoning models (content may be in 'reasoning' field)
        content = message.get("content") or ""
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        if not content and message.get("reasoning_details"):
            for detail in message["reasoning_details"]:
                if detail.get("text"):
                    content += detail["text"]

        input_tokens = usage.get("prompt_tokens", estimate_tokens(json.dumps(messages)))
        output_tokens = usage.get("completion_tokens", estimate_tokens(content))

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def _call_anthropic(self, messages: list, temperature: float,
                         max_tokens: int) -> dict:
        """Call Anthropic's Messages API."""
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity"
        }

        # Extract system from messages
        system_text = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append(m)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if system_text:
            payload["system"] = system_text

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if resp.status_code != 200:
            raise Exception(f"Anthropic API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        usage = data.get("usage", {})

        return {
            "content": data["content"][0]["text"],
            "input_tokens": usage.get("input_tokens", estimate_tokens(json.dumps(messages))),
            "output_tokens": usage.get("output_tokens", estimate_tokens(data["content"][0]["text"]))
        }

    def get_cost_summary(self) -> dict:
        """Get total cost summary for this client session."""
        return {
            "model": self.model,
            "total_calls": len(self.call_log),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_call": round(
                self.total_cost / max(len(self.call_log), 1), 4),
            "calls": self.call_log
        }


def create_client_from_config(config: dict) -> Optional[LLMClient]:
    """
    Create an LLMClient from a config dict (from dashboard API keys or .env).

    Config keys:
        api_key, base_url, model_name, provider
    """
    api_key = config.get("api_key") or config.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY", "")
    base_url = config.get("base_url") or config.get("LLM_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    model = config.get("model_name") or config.get("LLM_MODEL_NAME") or os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        return None
    if not base_url:
        base_url = "https://api.openai.com/v1"

    return LLMClient(api_key=api_key, base_url=base_url, model=model)


def create_client_from_env() -> Optional[LLMClient]:
    """Create client from environment variables or .env file."""
    # Try loading .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

    return create_client_from_config({})
