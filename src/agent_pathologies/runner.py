from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Sequence

from .budget import CostSpec, count_messages, count_tokens
from .client import LLMClient
from .analysis.exclusions import exclusion_reason
from .tasks.base import Scorer
from .types import ModelRole, Role, Trajectory, Turn


def _normalize_response(r):
    """Accept either an LLMResponse (new) or a bare string (legacy mocks)."""
    if hasattr(r, "text"):
        return r.text, getattr(r, "upstream_actual", None)
    return str(r), None


def _role_str(model_role: Any) -> str | None:
    """Normalize a model_role argument to its canonical string form
    regardless of whether it arrives as an enum value, an enum instance,
    or a string. Matters because cell_key is called both with the
    `ModelRole.INSTRUCT` enum (write time, from run.py) and with the
    JSON-decoded string `"instruct"` (read time, from existing_cell_keys
    canonicalizing a stored row)."""
    if model_role is None:
        return None
    val = getattr(model_role, "value", model_role)
    return str(val)


def cell_key(
    model: str,
    task_id: str,
    sweep_value: Any,
    seed: int | None,
    model_role: Any = None,
) -> str:
    """Compute the resumability key for a single trajectory cell.

    Includes `model_role` so that two siblings sharing the same `model` ID
    (e.g. DeepSeek V4-pro instruct vs. reasoning, distinguished only by a
    runtime `reasoning_config` flag) do not collide. Without this, the
    runner would skip a reasoning cell because the instruct sibling had
    already written a row with the same (model, task, sweep, seed)."""
    payload = json.dumps(
        {
            "model": model,
            "model_role": _role_str(model_role),
            "task": task_id,
            "sweep": sweep_value,
            "seed": seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def existing_cell_keys(path: Path) -> set[str]:
    """Read a JSONL log and return the set of cell-keys already attempted.
    Used by experiment runners to resume an interrupted sweep.

    Genuine model-behavior exclusions (refusal, truncation, unscorable,
    upstream_mismatch) count as attempted and are NOT re-sampled — the
    preregistration commits to reporting them as-is.

    Infrastructure exclusions (`provider_error:*`) are different: the trajectory
    never reached the model. We re-attempt those on resume so that a transient
    HTTP failure or credit-exhaustion 402 doesn't permanently corrupt a cell.

    Cell keys are recomputed from the stored row data (model, model_role,
    task_id, sweep_value, seed) rather than read from `extra.cell_key`,
    because rows written before the 2026-05-14 cell-key fix used an
    insufficient key that did not include `model_role` — V4-pro instruct
    and reasoning would have collided. Recomputing here disambiguates
    them retroactively.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = (row.get("exclusion_reason") or "")
            if reason.startswith("provider_error:"):
                continue  # re-attempt this cell — it never reached the model
            recomputed = cell_key(
                model=row.get("model"),
                task_id=row.get("task_id"),
                sweep_value=row.get("sweep_value"),
                seed=row.get("seed"),
                model_role=row.get("model_role"),
            )
            done.add(recomputed)
    return done


async def run_trajectory(
    client: LLMClient,
    turns: Sequence[Turn],
    *,
    task_id: str,
    task_name: str | None,
    experiment: str,
    correct_answer: str,
    scorer: Scorer,
    seed: int | None = None,
    temperature: float = 0.0,
    sweep_value: Any = None,
    max_tokens: int = 512,
    model_family: str | None = None,
    model_role: ModelRole | None = None,
    cost_spec: CostSpec | None = None,
    upstream_pinned: str | None = None,
    exploratory: bool = False,
) -> Trajectory:
    """Walk `turns`, generating any empty assistant turns. Score the last
    is_probe turn. Capture errors, count tokens, attach cost, flag exclusions.
    `upstream_pinned` is the OpenRouter upstream this client was *configured*
    to use; the actually-served upstream (per-call) is collected and the
    exclusion rule flags any mismatch."""

    materialized: list[Turn] = []
    history: list[dict[str, str]] = []
    input_tokens_total = 0
    output_tokens_total = 0
    error: str | None = None
    upstream_actuals: list[str] = []

    try:
        for t in turns:
            if t.role == Role.ASSISTANT and not t.content:
                input_tokens_total += count_messages(history, model=client.model)
                response = await client.complete(
                    history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                answer, upstream_actual = _normalize_response(response)
                if upstream_actual:
                    upstream_actuals.append(upstream_actual)
                output_tokens_total += count_tokens(answer, model=client.model)
                new_turn = t.model_copy(update={"content": answer})
                materialized.append(new_turn)
                history.append({"role": "assistant", "content": answer})
            else:
                materialized.append(t)
                history.append({"role": t.role.value, "content": t.content})
    except Exception as e:
        error = type(e).__name__ + ": " + str(e)[:200]

    probe = next((t for t in reversed(materialized) if t.is_probe), None)
    probe_answer = probe.content if probe and probe.content else None

    is_correct: bool | None = None
    if probe_answer is not None and not error:
        try:
            is_correct = scorer(probe_answer, correct_answer)
        except Exception as e:
            error = f"ScoringError: {type(e).__name__}: {str(e)[:120]}"

    cost = None
    if cost_spec is not None:
        cost = cost_spec.for_call(input_tokens_total, output_tokens_total)

    traj = Trajectory(
        task_id=task_id,
        task_name=task_name,
        experiment=experiment,
        model=client.model,
        provider=client.provider,
        model_family=model_family,
        model_role=model_role,
        seed=seed,
        temperature=temperature,
        turns=materialized,
        probe_answer=probe_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
        cost_usd=cost,
        error=error,
        sweep_value=sweep_value,
        extra={
            "cell_key": cell_key(client.model, task_id, sweep_value, seed, model_role=model_role),
            "upstream_pinned": upstream_pinned,
            "upstream_actual": upstream_actuals[-1] if upstream_actuals else None,
            "upstream_observed_all": upstream_actuals,
            "exploratory": exploratory,
        },
    )
    reason = exclusion_reason(traj, max_tokens=max_tokens)
    if reason:
        traj.excluded = True
        traj.exclusion_reason = reason
    return traj


def write_jsonl(traj: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(traj.model_dump_json() + "\n")


async def run_batch(
    coros: Sequence[Awaitable[Trajectory]], *, concurrency: int = 8
) -> list[Trajectory]:
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(c: Awaitable[Trajectory]) -> Trajectory:
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros])
