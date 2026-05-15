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
    OpenAI, OpenRouter, DeepSeek direct, etc.

    Two optional knobs:
      - `upstream_provider`: OpenRouter-only. Pins routing to one upstream host
        and disables fallback. The actually-served upstream is surfaced on the
        LLMResponse so the runner can flag mismatches.
      - `reasoning_config`: provider-specific reasoning toggle. For DeepSeek,
        Anthropic-compatible reasoning models, and OpenRouter we inject the
        provider's expected keys. None means default (provider's choice)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        provider: str = "openai_compat",
        timeout: float = 120.0,
        upstream_provider: str | None = None,
        reasoning_config: dict[str, Any] | None = None,
        thinking_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.upstream_provider = upstream_provider
        self.reasoning_config = reasoning_config
        # `thinking_config` is DeepSeek-direct's NATIVE toggle for the new
        # `deepseek-v4-flash` / `deepseek-v4-pro` model IDs. Per
        # https://api-docs.deepseek.com/api/create-chat-completion, the V4
        # family accepts `thinking: {"type": "enabled"|"disabled"}`. The
        # legacy `reasoning: {"enabled": bool}` format is silently ignored
        # for these IDs (the 2026-05-15 max_tokens diagnostic showed the
        # model defaults to thinking-mode-on regardless of `reasoning`
        # being set to false). For V4-pro instruct calls we MUST send
        # `thinking: {"type": "disabled"}` to actually disable CoT.
        self.thinking_config = thinking_config
        self._http = httpx.AsyncClient(timeout=timeout)

    def _inject_reasoning(self, payload: dict[str, Any]) -> None:
        """Provider-specific reasoning-toggle injection.

        Two formats are supported, controlled by which config was passed:
          - `reasoning_config` → `payload["reasoning"] = {...}` (legacy
            DeepSeek alias models, OpenRouter normalized format).
          - `thinking_config` → `payload["thinking"] = {...}` (DeepSeek-direct
            V4-family native format; required when calling
            `deepseek-v4-flash` / `deepseek-v4-pro` directly).
        Both are injected if both are set, in that order, so providers that
        only recognize one format silently ignore the other."""
        if self.reasoning_config and self.provider in ("openrouter", "deepseek_direct"):
            payload["reasoning"] = dict(self.reasoning_config)
        if self.thinking_config and self.provider == "deepseek_direct":
            payload["thinking"] = dict(self.thinking_config)

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=2, max=120))
    async def complete(self, messages, temperature=0.0, max_tokens=1024, seed=None):
        # 8 attempts × exponential backoff (2, 4, 8, 16, 32, 64, 120, 120 s) gives
        # ~6 minutes of total backoff before we surrender — long enough to ride
        # out an OpenRouter micro-storm (the 2026-05-14/15 ConnectError + 402
        # storms were typically 30-180s windows). Previously 4 attempts with
        # 30s max wait surrendered too quickly and yielded ~40% provider_error
        # rates on context_rot during stormy periods. Failures here are still
        # written as provider_error (retry-eligible) so a transient outage
        # never permanently corrupts a cell.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.provider == "openrouter" and self.upstream_provider:
            payload["provider"] = {
                "order": [self.upstream_provider],
                "allow_fallbacks": False,
            }
        self._inject_reasoning(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = await self._http.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        # Reasoning-model fallback. DeepSeek-direct's `deepseek-reasoner` returns
        # `reasoning_content` (CoT) and `content` (final answer) as sibling fields;
        # OpenRouter returns either `reasoning` (string) or `reasoning_details`
        # (list of structured blocks) depending on the upstream. When max_tokens
        # is exhausted by reasoning, `content` can be empty while the answer
        # (or at least its final sentence) sits at the tail of the reasoning
        # trace. We fall back to those fields so a clean `is_correct` score can
        # still be extracted via the scorer's regex. Confirmed empirically on
        # DeepSeek hardness-5 arithmetic at max_tokens=2048 (see test).
        if not text.strip():
            text = (msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or "")
            if not text.strip():
                details = msg.get("reasoning_details")
                if isinstance(details, list):
                    text = " ".join(
                        d.get("text", "") if isinstance(d, dict) else str(d)
                        for d in details
                    )
        # OpenRouter exposes the actual upstream in the response body
        # (`data["provider"]`) and as a header (`x-openrouter-provider`).
        # DeepSeek direct is its own upstream — we record the provider field.
        upstream_actual = (
            data.get("provider")
            or r.headers.get("x-openrouter-provider")
            or r.headers.get("X-OpenRouter-Provider")
            or (self.provider if self.provider == "deepseek_direct" else None)
        )
        return LLMResponse(text=text, upstream_actual=upstream_actual)


def get_client(
    provider: str,
    model: str,
    *,
    upstream_provider: str | None = None,
    reasoning_config: dict[str, Any] | None = None,
    thinking_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Provider factory.

    Recognized providers: mock | openrouter | deepseek_direct | anthropic |
    together | fireworks | vllm | ollama | openai.

    `upstream_provider` is meaningful for OpenRouter only (quietly ignored
    elsewhere). `reasoning_config` is meaningful for OpenRouter and
    deepseek_direct (legacy reasoning toggle). `thinking_config` is
    DeepSeek-direct's native V4-family toggle and MUST be used for direct
    calls to `deepseek-v4-flash` / `deepseek-v4-pro`; the legacy `reasoning`
    parameter is silently ignored for those IDs.
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
            reasoning_config=reasoning_config,
            thinking_config=thinking_config,
            **kwargs,
        )
    if p == "deepseek_direct":
        return OpenAICompatibleClient(
            model=model,
            base_url=kwargs.pop("base_url", "https://api.deepseek.com/v1"),
            api_key=kwargs.pop("api_key", os.environ.get("DEEPSEEK_API_KEY")),
            provider="deepseek_direct",
            reasoning_config=reasoning_config,
            thinking_config=thinking_config,
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
