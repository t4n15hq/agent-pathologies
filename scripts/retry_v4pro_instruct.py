"""Retry V4-pro INSTRUCT cells stuck on `provider_error` from the OR/Novita
storms. Routes through DeepSeek-direct instead of OR — same routing trick
that lifted V4-pro REASONING recovery to 100%.

Targets self_consistency only by default; can be extended to other axes with
--axis flag. Most provider_error cells on V4-pro instruct are concentrated
in self_consistency (323) because that's where the OR connectivity storms
hit hardest during the initial sweep.

Usage:
    python scripts/retry_v4pro_instruct.py --dry-run
    python scripts/retry_v4pro_instruct.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
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
    """Match V4-pro INSTRUCT cells stuck on infrastructure-side failures
    that DS-direct routing recovers cleanly:
      - provider_error  (OR/Novita ConnectError storm residue)
      - provider_empty_response  (200 OK with empty body)
      - empty_probe_answer + output_tokens=0  (analyzer reclassifies this
        to provider_empty_response, but the raw JSONL tag is still
        empty_probe_answer)
    """
    if row.get("excluded") is not True: return False
    if row.get("model_family") != "deepseek-v4-pro": return False
    if row.get("model_role") != "instruct": return False
    reason = row.get("exclusion_reason") or ""
    if reason.startswith("provider_error"): return True
    if reason == "provider_empty_response": return True
    if reason == "empty_probe_answer" and (row.get("output_tokens") or 0) == 0:
        return True
    return False


def build_turns_self_consistency(task, row: dict):
    task_seed = task_seed_from_id(row["task_id"])
    inst = task.sample(task_seed)
    return list(inst.setup_turns), inst


def build_turns_sycophancy(task, row: dict):
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    condition = sweep["condition"]
    gap = int(sweep["post_gap"])
    task_seed = task_seed_from_id(row["task_id"])
    inst = task.sample(task_seed)
    rng = random.Random(seed)
    turns: list[Turn] = list(inst.setup_turns)
    turns.extend(pushback(
        inst.correct_answer,
        condition,
        wrong_answer_override=inst.intuitive_wrong_answer,
    ))
    for _ in range(gap):
        turns.extend(filler_turn_pair("irrelevant", rng))
    turns.append(Turn(role=Role.USER,
                      content=f"Just to confirm: {inst.probe_question}"))
    turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
    return turns, inst


def build_turns_context_rot(task, row: dict):
    """Mirror `experiments/context_rot/run.py:87-90` exactly:
        turns = setup_turns + filler + user(probe_question) + assistant(empty, is_probe=True)
    The earlier (incorrect) version sliced `setup_turns[:-1]` then re-appended,
    which is a no-op for variable_tracking (its last setup turn is the empty
    assistant after the final update, NOT the probe). That meant NO turn was
    flagged is_probe, so `probe_answer` came back None on every retry."""
    sweep = row["sweep_value"]
    seed = int(row["seed"])
    kind = sweep["kind"]
    n_filler = int(sweep["n_filler"])
    task_seed = task_seed_from_id(row["task_id"])
    inst = task.sample(task_seed)
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

    # Two-pass: build the set of cells that ALREADY have a non-excluded
    # V4-pro-instruct row, then skip them on the failed pass. Without this,
    # re-running the script after a prior successful retry would waste
    # ~$0.16 of needless DS-direct calls on already-recovered cells.
    def cell_tuple(r: dict) -> tuple:
        sw = r.get("sweep_value")
        return (r.get("task_id"),
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
            if r.get("model_family") != "deepseek-v4-pro": continue
            if r.get("model_role") != "instruct": continue
            has_success.add(cell_tuple(r))

    failed = [json.loads(l) for l in path.open() if l.strip() and is_target(json.loads(l))]
    by_cell: dict[tuple, dict] = {}
    for r in failed:
        k = cell_tuple(r)
        if k in has_success: continue
        by_cell[k] = r
    unique = list(by_cell.values())

    print(f"\n[{axis}] V4-pro instruct provider_error rows: {len(failed)}  unique cells: {len(unique)}")
    if not unique:
        return 0
    if dry_run:
        return len(unique)

    builder = AXIS_BUILDERS[axis]
    spec = specs.get(("deepseek-v4-pro", "instruct"))
    if spec is None:
        print("  ERROR: no V4-pro instruct spec found in models.yaml")
        return 0

    coros = []
    for row in unique:
        # Route via DeepSeek-direct with thinking explicitly disabled. Use
        # the V4-family NATIVE parameter `thinking: {"type": "disabled"}`
        # — the legacy `reasoning: {"enabled": false}` is silently ignored
        # for `deepseek-v4-pro` and the model defaults to thinking-mode-on,
        # which exhausts max_tokens on long context_rot conversations.
        # Empirically diagnosed 2026-05-15.
        client = get_client(
            "deepseek_direct",
            "deepseek-v4-pro",
            thinking_config={"type": "disabled"},
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
                   default="self_consistency")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main(p.parse_args()))
