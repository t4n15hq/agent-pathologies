"""Sycophancy persistence: original Q -> A, pushback, gap turns, re-probe.
Three pushback conditions (wrong / correct / neutral) × four post-pushback
gaps. Paired across instruct/reasoning siblings."""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.config_loader import iter_run_specs, load_yaml, mock_run_specs
from agent_pathologies.conversation.pushback import pushback
from agent_pathologies.conversation.synthesizer import filler_turn_pair
from agent_pathologies.runner import (
    cell_key,
    existing_cell_keys,
    run_batch,
    run_trajectory,
    write_jsonl,
)
from agent_pathologies.tasks import get_task
from agent_pathologies.types import Role, Turn


async def main(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.config))["sycophancy"]
    if args.mock:
        specs = mock_run_specs()
    else:
        models_cfg = load_yaml(Path(args.models_config))
        specs = list(iter_run_specs(models_cfg))

    out = Path(args.out)
    if args.fresh and out.exists():
        out.unlink()
    done = existing_cell_keys(out)
    if done:
        print(f"resume mode: skipping {len(done)} already-completed cells")

    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    print(f"task: {task.name}")
    gaps = cfg["post_pushback_gaps"]
    conditions = cfg["pushback_conditions"]

    coros: list = []
    total_planned = 0
    skipped = 0
    for spec in specs:
        client = get_client(spec.provider, spec.model)
        for task_seed in range(cfg["n_tasks"]):
            inst = task.sample(task_seed)
            for condition in conditions:
                for gap in gaps:
                    sweep_value = {"condition": condition, "post_gap": gap}
                    seed = task_seed * 10_000 + gap * 100 + conditions.index(condition)
                    total_planned += 1
                    key = cell_key(spec.model, inst.task_id, sweep_value, seed)
                    if key in done:
                        skipped += 1
                        continue

                    rng = random.Random(seed)
                    turns = list(inst.setup_turns)  # system + Q + initial A
                    turns.extend(pushback(
                        inst.correct_answer,
                        condition,
                        wrong_answer_override=inst.intuitive_wrong_answer,
                    ))
                    for _ in range(gap):
                        turns.extend(filler_turn_pair("irrelevant", rng))
                    turns.append(Turn(
                        role=Role.USER,
                        content=f"Just to confirm: {inst.probe_question}",
                    ))
                    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))

                    async def _one(spec=spec, client=client, inst=inst, turns=turns,
                                   seed=seed, sweep_value=sweep_value):
                        tj = await run_trajectory(
                            client,
                            turns,
                            task_id=inst.task_id,
                            task_name=inst.task_name,
                            experiment="sycophancy",
                            correct_answer=inst.correct_answer,
                            scorer=inst.scorer,
                            seed=seed,
                            temperature=cfg["temperature"],
                            sweep_value=sweep_value,
                            max_tokens=cfg["max_tokens"],
                            model_family=spec.family,
                            model_role=spec.role,
                            cost_spec=spec.cost_spec,
                        )
                        write_jsonl(tj, out)
                        return tj

                    coros.append(_one())

    print(f"planned={total_planned}  skipped={skipped}  to_run={len(coros)}")
    if not coros:
        return
    results = await run_batch(coros, concurrency=args.concurrency)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"completed {len(results)} trajectories; this-session cost ≈ ${cost:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pivot_a.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", default="data/sycophancy.jsonl")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(p.parse_args()))
