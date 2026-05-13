from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Sequence

from .client import LLMClient
from .types import Role, Trajectory, Turn


async def run_trajectory(
    client: LLMClient,
    turns: Sequence[Turn],
    *,
    task_id: str,
    experiment: str,
    correct_answer: str | None = None,
    seed: int | None = None,
    temperature: float = 0.0,
    sweep_value: Any = None,
    max_tokens: int = 512,
) -> Trajectory:
    """Walk `turns`. Every assistant turn with empty content is generated.
    The last is_probe turn supplies the probe answer for scoring."""
    materialized: list[Turn] = []
    history: list[dict[str, str]] = []

    for t in turns:
        if t.role == Role.ASSISTANT and not t.content:
            answer = await client.complete(
                history, temperature=temperature, max_tokens=max_tokens, seed=seed
            )
            new_turn = t.model_copy(update={"content": answer})
            materialized.append(new_turn)
            history.append({"role": "assistant", "content": answer})
        else:
            materialized.append(t)
            history.append({"role": t.role.value, "content": t.content})

    probe = next((t for t in reversed(materialized) if t.is_probe), None)
    probe_answer = probe.content if probe else None
    is_correct: bool | None = None
    if correct_answer is not None and probe_answer is not None:
        is_correct = correct_answer.strip().lower() in probe_answer.strip().lower()

    return Trajectory(
        task_id=task_id,
        experiment=experiment,
        model=client.model,
        provider=client.provider,
        seed=seed,
        temperature=temperature,
        turns=materialized,
        probe_answer=probe_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
        sweep_value=sweep_value,
    )


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
