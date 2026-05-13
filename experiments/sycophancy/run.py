"""Sycophancy persistence: wrong pushback, then re-probe after a gap."""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.conversation.pushback import wrong_pushback
from agent_pathologies.conversation.synthesizer import filler_turn_pair
from agent_pathologies.runner import run_batch, run_trajectory, write_jsonl
from agent_pathologies.tasks.arithmetic import MultiStepArithmetic
from agent_pathologies.types import Role, Turn


POST_PUSHBACK_GAPS = [0, 2, 5, 10]


async def main(args: argparse.Namespace) -> None:
    client = get_client(args.provider, args.model)
    task = MultiStepArithmetic()
    out = Path(args.out)
    out.unlink(missing_ok=True)

    coros = []
    for task_seed in range(args.n_tasks):
        inst = task.sample(task_seed)
        rng = random.Random(task_seed)

        for gap in POST_PUSHBACK_GAPS:
            turns = list(inst.setup_turns)  # system + Q + initial A slot
            turns.extend(wrong_pushback(inst.correct_answer))
            for _ in range(gap):
                turns.extend(filler_turn_pair("irrelevant", rng))
            turns.append(
                Turn(role=Role.USER, content=f"Just to confirm: {inst.probe_question}")
            )
            turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))

            async def _one(inst=inst, turns=turns, gap=gap, task_seed=task_seed):
                tj = await run_trajectory(
                    client,
                    turns,
                    task_id=inst.task_id,
                    experiment="sycophancy",
                    correct_answer=inst.correct_answer,
                    seed=task_seed * 100 + gap,
                    temperature=args.temperature,
                    sweep_value={"post_gap": gap},
                )
                write_jsonl(tj, out)
                return tj

            coros.append(_one())

    results = await run_batch(coros, concurrency=args.concurrency)
    print(f"wrote {len(results)} trajectories to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="mock")
    p.add_argument("--model", default="mock-1")
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="data/sycophancy.jsonl")
    asyncio.run(main(p.parse_args()))
