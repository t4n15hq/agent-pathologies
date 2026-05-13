"""Self-consistency: rerun the same task at T=0 across replays, paired
across instruct/reasoning model siblings per PREREGISTRATION."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.config_loader import iter_run_specs, load_yaml, mock_run_specs
from agent_pathologies.runner import existing_cell_keys, cell_key, run_batch, run_trajectory, write_jsonl
from agent_pathologies.tasks.arithmetic import MultiStepArithmetic


async def main(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.config))["self_consistency"]
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

    task = MultiStepArithmetic(hardness=cfg.get("hardness", 1))

    total_planned = 0
    skipped = 0
    coros = []
    for spec in specs:
        client = get_client(spec.provider, spec.model)
        for task_seed in range(cfg["n_tasks"]):
            inst = task.sample(task_seed)
            for repeat in range(cfg["n_repeats"]):
                seed = task_seed * 10_000 + repeat
                total_planned += 1
                key = cell_key(spec.model, inst.task_id, repeat, seed)
                if key in done:
                    skipped += 1
                    continue

                async def _one(spec=spec, client=client, inst=inst, seed=seed, repeat=repeat):
                    tj = await run_trajectory(
                        client,
                        inst.setup_turns,
                        task_id=inst.task_id,
                        task_name=inst.task_name,
                        experiment="self_consistency",
                        correct_answer=inst.correct_answer,
                        scorer=inst.scorer,
                        seed=seed,
                        temperature=cfg["temperature"],
                        sweep_value=repeat,
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
        print("nothing to run")
        return
    results = await run_batch(coros, concurrency=args.concurrency)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"completed {len(results)} trajectories; this-session cost ≈ ${cost:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pivot_a.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--mock", action="store_true",
                   help="Use mock model specs (no API calls)")
    p.add_argument("--out", default="data/self_consistency.jsonl")
    p.add_argument("--fresh", action="store_true", help="Delete existing JSONL first")
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(p.parse_args()))
