"""Focused retry of Qwen `provider_error` cells via OpenRouter.

Qwen has no first-party API in our setup; the only route is OR (Novita /
Alibaba / Nebius / AtlasCloud). When the OR budget is tight, we don't
want generic context_rot resume to fan out attempts across V4-pro cells
that have alternative DS-direct routes. This script targets only Qwen
families and only `provider_error` exclusions.

Skips cells that already have a non-excluded row, so it's safe to
re-launch after partial progress.

Usage:
    python scripts/retry_qwen_provider_error.py --dry-run
    python scripts/retry_qwen_provider_error.py --axis context_rot --concurrency 12
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


def task_seed_from_id(task_id: str) -> int:
    m = re.search(r"(\d+)$", task_id)
    if not m:
        raise ValueError(f"can't extract seed from task_id={task_id!r}")
    return int(m.group(1))


def is_target(row: dict) -> bool:
    """Match Qwen cells stuck on provider_error OR empty_probe_answer.
    The latter class now includes cells that an earlier buggy
    context_rot retry script reconstructed without an `is_probe=True`
    assistant turn — those wrote rows that look like empty probes but
    are actually fixable by re-running with the corrected reconstruction.
    Fixed 2026-05-15."""
    fam = row.get("model_family") or ""
    if row.get("excluded") is not True: return False
    if not fam.startswith("qwen"): return False
    reason = row.get("exclusion_reason") or ""
    return reason.startswith("provider_error") or reason == "empty_probe_answer"


def build_turns_self_consistency(task, row):
    inst = task.sample(task_seed_from_id(row["task_id"]))
    return list(inst.setup_turns), inst


def build_turns_sycophancy(task, row):
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    inst = task.sample(task_seed_from_id(row["task_id"]))
    rng = random.Random(seed)
    turns: list[Turn] = list(inst.setup_turns)
    turns.extend(pushback(inst.correct_answer, sweep["condition"],
                          wrong_answer_override=inst.intuitive_wrong_answer))
    for _ in range(int(sweep["post_gap"])):
        turns.extend(filler_turn_pair("irrelevant", rng))
    turns.append(Turn(role=Role.USER,
                      content=f"Just to confirm: {inst.probe_question}"))
    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
    return turns, inst


def build_turns_context_rot(task, row):
    """Mirror `experiments/context_rot/run.py:87-90` exactly. The empty
    assistant probe MUST be flagged is_probe=True. Fixed 2026-05-15."""
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    inst = task.sample(task_seed_from_id(row["task_id"]))
    rng = random.Random(seed)
    turns: list[Turn] = list(inst.setup_turns)
    turns.extend(build_filler_block(sweep["kind"], int(sweep["n_filler"]), rng))
    turns.append(Turn(role=Role.USER, content=inst.probe_question))
    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
    return turns, inst


AXIS_BUILDERS = {
    "self_consistency": build_turns_self_consistency,
    "sycophancy":       build_turns_sycophancy,
    "context_rot":      build_turns_context_rot,
}


async def retry_axis(axis: str, *, concurrency: int, dry_run: bool, max_tokens: int) -> int:
    cfg = load_yaml(Path("configs/pivot_a.yaml"))[axis]
    models_cfg = load_yaml(Path("configs/models.yaml"))
    specs = {(s.family, s.role.value): s for s in iter_run_specs(
        models_cfg, include_anchors=False)}
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    path = Path(f"data/{axis}.jsonl")

    def cell_tuple(r):
        sw = r.get("sweep_value")
        return (r.get("model_family"), r.get("model_role"), r.get("task_id"),
                json.dumps(sw, sort_keys=True, default=str) if sw is not None else None,
                r.get("seed"))

    has_success: set[tuple] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get("excluded"): continue
            fam = r.get("model_family") or ""
            if not fam.startswith("qwen"): continue
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

    from collections import Counter
    by = Counter()
    for r in failed:
        by[(r.get("model_family"), r.get("model_role"))] += 1
    print(f"\n[{axis}] Qwen provider_error unique cells to retry: {len(failed)}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")

    if not failed or dry_run:
        return len(failed)

    builder = AXIS_BUILDERS[axis]
    coros = []
    for row in failed:
        key = (row.get("model_family"), row.get("model_role"))
        spec = specs.get(key)
        if spec is None:
            print(f"  warn: no spec for {key}")
            continue
        client = get_client(
            spec.provider, spec.model,
            upstream_provider=spec.upstream_provider,
            reasoning_config=spec.reasoning_config,
        )
        try:
            turns, inst = builder(task, row)
        except Exception as e:
            print(f"  warn: rebuild failed: {type(e).__name__}: {e}")
            continue

        async def _one(client=client, turns=turns, row=row, inst=inst, spec=spec):
            tj = await run_trajectory(
                client, turns,
                task_id=row["task_id"],
                task_name=row.get("task_name"),
                experiment=axis,
                correct_answer=row["correct_answer"],
                scorer=inst.scorer,
                upstream_pinned=spec.upstream_provider,
                exploratory=spec.exploratory,
                seed=row.get("seed"),
                temperature=cfg["temperature"],
                sweep_value=row.get("sweep_value"),
                max_tokens=max_tokens,
                model_family=spec.family,
                model_role=spec.role,
                cost_spec=spec.cost_spec,
            )
            write_jsonl(tj, path)
            return tj

        coros.append(_one())

    print(f"  launching {len(coros)} retries at concurrency={concurrency}, max_tokens={max_tokens}")
    results = await run_batch(coros, concurrency=concurrency)
    recovered = sum(1 for t in results if not t.excluded)
    still_failed = sum(1 for t in results if t.excluded)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"  [{axis}] recovered={recovered}  still_failed={still_failed}  cost≈${cost:.4f}")
    return recovered


async def main(args):
    from dotenv import load_dotenv
    load_dotenv()
    axes = ["self_consistency", "sycophancy", "context_rot"] if args.axis == "all" else [args.axis]
    total = 0
    for ax in axes:
        total += await retry_axis(ax, concurrency=args.concurrency, dry_run=args.dry_run,
                                    max_tokens=args.max_tokens)
    print(f"\nTOTAL: {'planned' if args.dry_run else 'recovered'} {total} cells.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["all", "self_consistency", "sycophancy", "context_rot"],
                   default="context_rot")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main(p.parse_args()))
