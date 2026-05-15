"""Retry V4-pro REASONING cells stuck on `provider_error` from the original
OR/Novita sweep, via DeepSeek-direct.

Mirror of `retry_v4pro_instruct.py` but targets the reasoning role with
reasoning_config={"enabled": True}. The DS empties retry doesn't catch
these because its filter only matches empty/truncated exclusion classes;
`provider_error` rows escape it.

Concentrated on self_consistency (182 cells). Sycophancy and context_rot
have far fewer V4-pro reasoning provider_error cells, but the script
checks all three axes for completeness.

Usage:
    python scripts/retry_v4pro_reasoning_provider_error.py --dry-run
    python scripts/retry_v4pro_reasoning_provider_error.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from agent_pathologies.client import get_client
from agent_pathologies.config_loader import iter_run_specs, load_yaml
from agent_pathologies.conversation.pushback import pushback
from agent_pathologies.conversation.synthesizer import build_filler_block, filler_turn_pair
from agent_pathologies.runner import run_batch, run_trajectory, write_jsonl
from agent_pathologies.tasks import get_task
from agent_pathologies.types import Role, Turn


NEW_MAX_TOKENS = 8192


def task_seed_from_id(task_id: str) -> int:
    m = re.search(r"(\d+)$", task_id)
    if not m:
        raise ValueError(f"can't extract seed from task_id={task_id!r}")
    return int(m.group(1))


def is_target(row: dict) -> bool:
    """V4-pro REASONING cells stuck on provider_error, regardless of axis."""
    return (
        row.get("excluded") is True
        and row.get("model_family") == "deepseek-v4-pro"
        and row.get("model_role") == "reasoning"
        and (row.get("exclusion_reason") or "").startswith("provider_error")
    )


def build_turns_self_consistency(task, row: dict):
    inst = task.sample(task_seed_from_id(row["task_id"]))
    return list(inst.setup_turns), inst


def build_turns_sycophancy(task, row: dict):
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    condition = sweep["condition"]
    gap = int(sweep["post_gap"])
    inst = task.sample(task_seed_from_id(row["task_id"]))
    rng = random.Random(seed)
    turns: list[Turn] = list(inst.setup_turns)
    turns.extend(pushback(
        inst.correct_answer, condition,
        wrong_answer_override=inst.intuitive_wrong_answer,
    ))
    for _ in range(gap):
        turns.extend(filler_turn_pair("irrelevant", rng))
    turns.append(Turn(role=Role.USER,
                      content=f"Just to confirm: {inst.probe_question}"))
    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
    return turns, inst


def build_turns_context_rot(task, row: dict):
    """Mirror `experiments/context_rot/run.py:87-90` exactly. The empty
    assistant probe MUST be flagged is_probe=True or the runner won't
    score the trajectory. Fixed 2026-05-15."""
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    kind = sweep["kind"]
    n_filler = int(sweep["n_filler"])
    inst = task.sample(task_seed_from_id(row["task_id"]))
    rng = random.Random(seed)
    turns: list[Turn] = list(inst.setup_turns)
    turns.extend(build_filler_block(kind, n_filler, rng))
    turns.append(Turn(role=Role.USER, content=inst.probe_question))
    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
    return turns, inst


AXIS_BUILDERS = {
    "self_consistency": build_turns_self_consistency,
    "sycophancy":       build_turns_sycophancy,
    "context_rot":      build_turns_context_rot,
}


async def retry_axis(axis: str, *, concurrency: int, dry_run: bool) -> int:
    cfg = load_yaml(Path("configs/pivot_a.yaml"))[axis]
    models_cfg = load_yaml(Path("configs/models.yaml"))
    specs = {(s.family, s.role.value): s for s in iter_run_specs(
        models_cfg, include_anchors=False)}
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    path = Path(f"data/{axis}.jsonl")

    # Two-pass scan: skip cells where a non-excluded V4-pro-reasoning row
    # already exists (same cell, different attempt succeeded).
    def cell_tuple(r: dict) -> tuple:
        sw = r.get("sweep_value")
        sw_key = json.dumps(sw, sort_keys=True, default=str) if sw is not None else None
        return (r.get("task_id"), sw_key, r.get("seed"))

    has_success: set[tuple] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get("excluded"): continue
            if r.get("model_family") != "deepseek-v4-pro": continue
            if r.get("model_role") != "reasoning": continue
            has_success.add(cell_tuple(r))

    seen: set[tuple] = set()
    failed: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if not is_target(r): continue
            t = cell_tuple(r)
            if t in has_success or t in seen: continue
            seen.add(t)
            failed.append(r)

    print(f"\n[{axis}] V4-pro reasoning provider_error rows to retry: {len(failed)}")
    if not failed: return 0
    if dry_run: return len(failed)

    builder = AXIS_BUILDERS[axis]
    spec = specs.get(("deepseek-v4-pro", "reasoning"))
    if spec is None:
        print("  ERROR: no V4-pro reasoning spec")
        return 0

    coros = []
    for row in failed:
        client = get_client(
            "deepseek_direct", "deepseek-v4-pro",
            reasoning_config={"enabled": True},
        )
        try:
            turns, inst = builder(task, row)
        except Exception as e:
            print(f"  warn: rebuild failed for {row.get('task_id')}: {type(e).__name__}: {e}")
            continue

        async def _one(client=client, turns=turns, row=row, inst=inst, spec=spec):
            tj = await run_trajectory(
                client, turns,
                task_id=row["task_id"],
                task_name=row.get("task_name"),
                experiment=axis,
                correct_answer=row["correct_answer"],
                scorer=inst.scorer,
                upstream_pinned="deepseek_direct",
                exploratory=spec.exploratory,
                seed=row.get("seed"),
                temperature=cfg["temperature"],
                sweep_value=row.get("sweep_value"),
                max_tokens=NEW_MAX_TOKENS,
                model_family=spec.family,
                model_role=spec.role,
                cost_spec=spec.cost_spec,
            )
            write_jsonl(tj, path)
            return tj

        coros.append(_one())

    print(f"  launching {len(coros)} retries at concurrency={concurrency}")
    results = await run_batch(coros, concurrency=concurrency)
    recovered = sum(1 for t in results if not t.excluded)
    still_failed = sum(1 for t in results if t.excluded)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"  [{axis}] recovered={recovered}  still_failed={still_failed}  cost≈${cost:.4f}")
    return recovered


async def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    axes = ["self_consistency", "sycophancy", "context_rot"] if args.axis == "all" else [args.axis]
    total = 0
    for ax in axes:
        total += await retry_axis(ax, concurrency=args.concurrency, dry_run=args.dry_run)
    print(f"\nTOTAL: {'planned' if args.dry_run else 'recovered'} {total} cells.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["all", "self_consistency", "sycophancy", "context_rot"],
                   default="all")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main(p.parse_args()))
