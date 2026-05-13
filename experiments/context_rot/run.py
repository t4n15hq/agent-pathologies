"""Context rot: needle planted at turn 1, k filler pairs inserted, then
probed. Three filler kinds: irrelevant, related, token_matched.
Paired across instruct/reasoning siblings."""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.config_loader import iter_run_specs, load_yaml, mock_run_specs
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
    cfg = load_yaml(Path(args.config))["context_rot"]
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
    filler_counts = cfg["filler_counts"]
    filler_kinds = cfg["filler_kinds"]

    coros: list = []
    total_planned = 0
    skipped = 0
    for spec in specs:
        client = get_client(spec.provider, spec.model)
        for task_seed in range(cfg["n_tasks"]):
            inst = task.sample(task_seed)
            rng_seed_base = task_seed * 100_000
            for kind in filler_kinds:
                for k in filler_counts:
                    sweep_value = {"n_filler": k, "kind": kind}
                    seed = rng_seed_base + k * 10 + filler_kinds.index(kind)
                    total_planned += 1
                    key = cell_key(spec.model, inst.task_id, sweep_value, seed)
                    if key in done:
                        skipped += 1
                        continue

                    rng = random.Random(seed)
                    turns = list(inst.setup_turns)
                    for _ in range(k):
                        turns.extend(filler_turn_pair(kind, rng))
                    turns.append(Turn(role=Role.USER, content=inst.probe_question))
                    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))

                    async def _one(spec=spec, client=client, inst=inst, turns=turns,
                                   seed=seed, sweep_value=sweep_value):
                        tj = await run_trajectory(
                            client,
                            turns,
                            task_id=inst.task_id,
                            task_name=inst.task_name,
                            experiment="context_rot",
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
    p.add_argument("--out", default="data/context_rot.jsonl")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(p.parse_args()))
