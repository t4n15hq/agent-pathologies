from __future__ import annotations

import asyncio
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMResponse:
    """Result of a single client.complete() call. Returning a structured
    object (rather than a bare str) lets clients surface metadata like the
    upstream host that served the request — needed for OpenRouter routing
    pinning checks. The metadata is per-call, so concurrent coroutines
    sharing a client don't race on shared instance state."""

    text: str
    upstream_actual: str | None = None


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
    ) -> LLMResponse: ...


class MockClient(LLMClient):
    """Deterministic-ish stand-in. Lets the whole pipeline run with no API key."""

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
            return LLMResponse("Ready.")
        text = last_user["content"]

        m = re.search(
            r"(?:actually|no,?\s+it'?s|the (?:answer|correct)(?: is| should be))\s+(-?\d+|[A-D])\b",
            text,
            re.I,
        )
        if m and rng.random() < self.sycophancy_rate:
            return LLMResponse(f"You're right, I apologize — the answer is {m.group(1)}.")

        expr = re.search(r"compute\s+(.+?)(?:\.|$)", text, re.I)
        if expr:
            try:
                value = eval(expr.group(1), {"__builtins__": {}}, {})  # noqa: S307
                if rng.random() < self.error_rate:
                    value = int(value) + rng.choice([-2, -1, 1, 2])
                return LLMResponse(f"The answer is {value}.")
            except Exception:
                pass

        if re.search(r"current value of\s+x", text, re.I):
            x = self._evaluate_variable_track(messages)
            if x is not None:
                if rng.random() < self.error_rate:
                    x += rng.choice([-2, -1, 1, 2])
                return LLMResponse(f"The answer is {x}.")

        if "```python" in text:
            return LLMResponse("The answer is 0.")

        if "the answer is n" in text.lower() or "respond with exactly" in text.lower():
            stable = abs(hash(text)) % 100
            return LLMResponse(f"The answer is {stable}.")

        history_text = "\n".join(m["content"] for m in messages)
        facts: dict[str, str] = {}
        for m in re.finditer(
            r"remember:\s*([^=\n]+?)\s*=\s*([^\n.]+)", history_text, re.I
        ):
            facts[m.group(1).strip().lower()] = m.group(2).strip()
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
                return LLMResponse(f"{queried} is {val}.")

        return LLMResponse("Acknowledged.")

    @staticmethod
    def _evaluate_variable_track(messages):
        """Best-effort: replay the variable updates seen in chat history."""
        x = None
        for m in messages:
            content = m.get("content", "")
            init = re.search(r"x\s*=\s*(-?\d+)", content)
            if init and x is None:
                x = int(init.group(1))
                continue
            if x is None:
                continue
            add = re.search(r"add\s+(\d+)\s+to\s+x", content, re.I)
            sub = re.search(r"subtract\s+(\d+)\s+from\s+x", content, re.I)
            mul = re.search(r"multiply\s+x\s+by\s+(\d+)", content, re.I)
            if add:
                x += int(add.group(1))
            elif sub:
                x -= int(sub.group(1))
            elif mul:
                x *= int(mul.group(1))
        return x


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 **_ignored: Any) -> None:
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
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return LLMResponse(text)


class OpenAICompatibleClient(LLMClient):
    """OpenAI /chat/completions wire format — Together, Fireworks, vLLM, Ollama,
    OpenAI, OpenRouter. When `upstream_provider` is set (OpenRouter only), we
    inject the provider routing config so OpenRouter pins this exact upstream
    host and refuses to fall back. The actual upstream host that served the
    request is surfaced on the LLMResponse for downstream verification."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        provider: str = "openai_compat",
        timeout: float = 120.0,
        upstream_provider: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.upstream_provider = upstream_provider
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
        # OpenRouter provider pinning — only meaningful when talking to OpenRouter.
        if self.provider == "openrouter" and self.upstream_provider:
            payload["provider"] = {
                "order": [self.upstream_provider],
                "allow_fallbacks": False,
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = await self._http.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        # OpenRouter exposes the actual upstream both in the response body
        # (`data["provider"]`) and as a response header (`x-openrouter-provider`).
        # We prefer the body and fall back to the header.
        upstream_actual = (
            data.get("provider")
            or r.headers.get("x-openrouter-provider")
            or r.headers.get("X-OpenRouter-Provider")
        )
        return LLMResponse(text=text, upstream_actual=upstream_actual)


def get_client(
    provider: str,
    model: str,
    *,
    upstream_provider: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Provider factory.

    Recognized providers: mock | openrouter | anthropic | together | fireworks
    | vllm | ollama | openai.

    `upstream_provider` is meaningful for OpenRouter (and quietly ignored by
    other providers — clients that don't proxy to upstream hosts).
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
            upstream_provider=upstream_provider,
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
