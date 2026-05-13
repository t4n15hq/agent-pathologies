from __future__ import annotations

import asyncio
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> str: ...


class MockClient(LLMClient):
    """Deterministic-ish stand-in. Lets the whole pipeline run with no API key.

    Behaviors that make the smoke tests informative:
      - Solves arithmetic prompts via a sandboxed eval.
      - Recalls 'Remember: X = Y' facts when later asked 'What is X?'.
      - Flips on wrong pushback ('the answer is actually Z') with probability
        `sycophancy_rate`.
    """

    provider = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        error_rate: float = 0.0,
        sycophancy_rate: float = 0.5,
    ) -> None:
        self.model = model
        self.error_rate = error_rate
        self.sycophancy_rate = sycophancy_rate

    async def complete(self, messages, temperature=0.0, max_tokens=1024, seed=None):
        await asyncio.sleep(0)
        rng = random.Random(seed if seed is not None else hash(str(messages)))
        last_user = next(
            (m for m in reversed(messages) if m["role"] == "user"), None
        )
        if not last_user:
            return "Ready."
        text = last_user["content"]

        m = re.search(
            r"(?:actually|no,?\s+it'?s|the (?:answer|correct)(?: is| should be))\s+([^\s.!?]+)",
            text,
            re.I,
        )
        if m and rng.random() < self.sycophancy_rate:
            return f"You're right, I apologize — the answer is {m.group(1)}."

        expr = re.search(r"compute\s+(.+?)(?:\.|$)", text, re.I)
        if expr:
            try:
                value = eval(expr.group(1), {"__builtins__": {}}, {})  # noqa: S307
                if rng.random() < self.error_rate:
                    value = int(value) + rng.choice([-2, -1, 1, 2])
                return f"The answer is {value}."
            except Exception:
                pass

        history_text = "\n".join(m["content"] for m in messages)
        facts: dict[str, str] = {}
        # Single-fact 'Remember: KEY = VALUE' form.
        for m in re.finditer(
            r"remember:\s*([^=\n]+?)\s*=\s*([^\n.]+)", history_text, re.I
        ):
            facts[m.group(1).strip().lower()] = m.group(2).strip()
        # Multi-fact bullet form: '- KEY = VALUE'.
        for m in re.finditer(
            r"^\s*[-*]\s*([^=\n]+?)\s*=\s*([^\n]+)$", history_text, re.M
        ):
            facts[m.group(1).strip().lower()] = m.group(2).strip()

        probe_m = re.search(r"what\s+(?:is|was)\s+(.+?)\??\s*$", text, re.I)
        if probe_m and facts:
            queried = probe_m.group(1).strip().lower().rstrip("?")
            if queried in facts:
                val = facts[queried]
                if rng.random() < self.error_rate:
                    val = val + "_wrong"
                return f"{queried} is {val}."

        return "Acknowledged."


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e
        self.model = model
        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def complete(self, messages, temperature=0.0, max_tokens=1024, seed=None):
        system = ""
        msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]).strip()
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        resp = await self._client.messages.create(
            model=self.model,
            system=system if system else self._anthropic.NOT_GIVEN,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


class OpenAICompatibleClient(LLMClient):
    """Talks to anything speaking OpenAI's /chat/completions wire format:
    Together AI, Fireworks, vLLM, Ollama, OpenAI itself."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        provider: str = "openai_compat",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self._http = httpx.AsyncClient(timeout=timeout)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def complete(self, messages, temperature=0.0, max_tokens=1024, seed=None):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = await self._http.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""


def get_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """Provider factory.

    Recognized providers: mock | openrouter | anthropic | together | fireworks
    | vllm | ollama | openai.
    """
    p = provider.lower()
    if p == "mock":
        return MockClient(model=model, **kwargs)
    if p == "openrouter":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "https://openrouter.ai/api/v1"),
            api_key=kwargs.pop("api_key", os.environ.get("OPENROUTER_API_KEY")),
            provider="openrouter",
            **kwargs,
        )
    if p == "anthropic":
        return AnthropicClient(model=model, **kwargs)
    if p == "together":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "https://api.together.xyz/v1"),
            api_key=kwargs.pop("api_key", os.environ.get("TOGETHER_API_KEY")),
            provider="together",
            **kwargs,
        )
    if p == "fireworks":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "https://api.fireworks.ai/inference/v1"),
            api_key=kwargs.pop("api_key", os.environ.get("FIREWORKS_API_KEY")),
            provider="fireworks",
            **kwargs,
        )
    if p == "vllm":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "http://localhost:8000/v1"),
            api_key=kwargs.pop("api_key", "EMPTY"),
            provider="vllm",
            **kwargs,
        )
    if p == "ollama":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "http://localhost:11434/v1"),
            api_key=kwargs.pop("api_key", "ollama"),
            provider="ollama",
            **kwargs,
        )
    if p == "openai":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "https://api.openai.com/v1"),
            api_key=kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY")),
            provider="openai",
            **kwargs,
        )
    raise ValueError(f"Unknown provider: {provider}")
