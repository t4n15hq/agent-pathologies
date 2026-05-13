"""Self-consistency: rerun the same task at T=0 many times, log everything."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.runner import run_batch, run_trajectory, write_jsonl
from agent_pathologies.tasks.arithmetic import MultiStepArithmetic


async def main(args: argparse.Namespace) -> None:
    client = get_client(args.provider, args.model)
    task = MultiStepArithmetic()
    out = Path(args.out)
    out.unlink(missing_ok=True)

    coros = []
    for task_seed in range(args.n_tasks):
        inst = task.sample(task_seed)
        for repeat in range(args.n_repeats):
            seed = task_seed * 10_000 + repeat

            async def _one(inst=inst, seed=seed, repeat=repeat):
                tj = await run_trajectory(
                    client,
                    inst.setup_turns,
                    task_id=inst.task_id,
                    experiment="self_consistency",
                    correct_answer=inst.correct_answer,
                    seed=seed,
                    temperature=args.temperature,
                    sweep_value=repeat,
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
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="data/self_consistency.jsonl")
    asyncio.run(main(p.parse_args()))
