"""Token counting + cost estimation. Keeps the user from accidentally
spending $$$ on a sweep. Used by runner.py for live tracking and by
scripts/estimate_cost.py for pre-sweep dry runs."""

from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass
class CostSpec:
    price_in_per_m: float
    price_out_per_m: float

    def for_call(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.price_in_per_m
            + output_tokens / 1_000_000 * self.price_out_per_m
        )


@functools.lru_cache(maxsize=8)
def _encoder_for(model: str):
    """Best-effort tokenizer. We don't have perfect tokenizers for every
    open-weight model, so fall back to cl100k_base (GPT-4 tokenizer) which
    is a reasonable proxy. Returns None only if tiktoken itself is missing."""
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    if not text:
        return 0
    enc = _encoder_for(model)
    if enc is None:
        return max(1, len(text) // 4)  # rough fallback
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_messages(messages: list[dict[str, str]], model: str = "gpt-4") -> int:
    return sum(
        count_tokens(m.get("content", ""), model) + 4  # role-overhead estimate
        for m in messages
    ) + 2


def estimate_sweep_cost(
    n_calls: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    spec: CostSpec,
) -> float:
    return n_calls * spec.for_call(avg_input_tokens, avg_output_tokens)
