"""Retry failed DeepSeek-reasoning trajectories with raised max_tokens.

Root cause discovered 2026-05-15: `max_tokens=2048` was set 16× below the
DeepSeek-recommended default (32K). On hardness-5 arithmetic the reasoning
model burns the entire budget on `reasoning_content` and emits empty
`content` — recorded as `empty_probe_answer` or `provider_empty_response`.

This script targets ONLY those failure rows on DeepSeek reasoning cells,
reconstructs the exact same prompt + sweep parameters, re-runs with
`max_tokens=8192`, and appends new rows to the same JSONL. Successful
retries lift analyzable count; persistent failures stay marked as
behavioral exclusions in the second pass too.

The client.py parser was also updated to read `reasoning_content` /
`reasoning_details` as fallback when `content` is empty, providing a
defense-in-depth recovery path for any cells where the model still
exhausts the higher budget.

Usage:
    python scripts/retry_deepseek_empties.py                      # all axes
    python scripts/retry_deepseek_empties.py --axis sycophancy    # one axis
    python scripts/retry_deepseek_empties.py --dry-run            # plan only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Iterator

import re

from agent_pathologies.client import get_client
from agent_pathologies.config_loader import iter_run_specs, load_yaml
from agent_pathologies.conversation.pushback import pushback
from agent_pathologies.conversation.synthesizer import build_filler_block, filler_turn_pair
from agent_pathologies.runner import run_batch, run_trajectory, write_jsonl
from agent_pathologies.tasks import get_task
from agent_pathologies.types import Role, Turn


RETRY_REASONS = {
    "empty_probe_answer",
    "provider_empty_response",
    "truncated_at_max_tokens",
    "unscorable_answer",
}

# Default; overridable via --max-tokens. Context_rot reasoning cells need
# 16K because their input conversations carry 20-40 filler turn-pairs and
# the model still needs ~3,500 reasoning tokens + a final answer.
NEW_MAX_TOKENS = 8192


def task_seed_from_id(task_id: str) -> int:
    """Task IDs follow the pattern `<prefix>-...-<seed>` where seed is the
    final numeric segment. E.g. `arith-h5-25` → 25, `vartrack-n20-12` → 12,
    `crt-7` → 7."""
    m = re.search(r"(\d+)$", task_id)
    if not m:
        raise ValueError(f"can't extract seed from task_id={task_id!r}")
    return int(m.group(1))


def iter_failed(path: Path) -> Iterator[dict]:
    """Yield JSONL rows that are DeepSeek-reasoning failures eligible for retry,
    SKIPPING cells where a successful (non-excluded) retry already exists.

    Two-pass:
      1. Scan all rows; build the set of (family, role, task, sweep, seed)
         tuples that already have at least one non-excluded row.
      2. Scan again; yield matching failed rows whose tuple is NOT in that set
         (and dedup at the tuple level: yield one row per cell).

    This makes the script safe to re-launch after partial progress without
    re-doing successful retries. On the 2026-05-15 retry, ~1,085 self-cons
    cells were already recovered; without this filter a restart would
    re-attempt them and burn ~$3 of needless DS-direct calls."""
    if not path.exists():
        return

    def _cell_tuple(r: dict) -> tuple:
        # Match the analysis-layer cell-key fields (model_family, role, task,
        # sweep, seed) so OR-original + DS-direct-retry collapse together.
        sw = r.get("sweep_value")
        sw_key = json.dumps(sw, sort_keys=True, default=str) if sw is not None else None
        return (
            r.get("model_family"),
            r.get("model_role"),
            r.get("task_id"),
            sw_key,
            r.get("seed"),
        )

    has_success: set[tuple] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("excluded"):
                continue
            fam = r.get("model_family") or ""
            role = r.get("model_role") or ""
            if not fam.startswith("deepseek") or role != "reasoning":
                continue
            has_success.add(_cell_tuple(r))

    seen_failed: set[tuple] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("excluded"):
                continue
            if (r.get("exclusion_reason") or "") not in RETRY_REASONS:
                continue
            fam = r.get("model_family") or ""
            role = r.get("model_role") or ""
            if not fam.startswith("deepseek") or role != "reasoning":
                continue
            t = _cell_tuple(r)
            if t in has_success or t in seen_failed:
                continue
            seen_failed.add(t)
            yield r


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
    Earlier (incorrect) version sliced `setup_turns[:-1]`, which is a no-op
    for variable_tracking (its last setup turn is the empty assistant after
    the final update, NOT the probe). That meant NO turn was flagged
    is_probe → `probe_answer=None` on every retry. Fixed 2026-05-15."""
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


async def retry_axis(axis: str, *, concurrency: int, dry_run: bool, max_tokens: int = NEW_MAX_TOKENS) -> int:
    cfg = load_yaml(Path("configs/pivot_a.yaml"))[axis]
    models_cfg = load_yaml(Path("configs/models.yaml"))
    specs = {(s.family, s.role.value): s for s in iter_run_specs(
        models_cfg, include_anchors=False)}

    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    path = Path(f"data/{axis}.jsonl")
    failed = list(iter_failed(path))

    print(f"\n[{axis}] failed DeepSeek-reasoning rows eligible for retry: {len(failed)}")
    if not failed:
        return 0

    builder = AXIS_BUILDERS[axis]
    by_cell: dict[tuple[str, str], int] = {}
    for r in failed:
        by_cell[(r.get("model_family"), r.get("model_role"))] = (
            by_cell.get((r.get("model_family"), r.get("model_role")), 0) + 1
        )
    for k, v in sorted(by_cell.items()):
        print(f"  {k}: {v} cells to retry")

    if dry_run:
        return len(failed)

    coros = []
    for row in failed:
        key = (row.get("model_family"), row.get("model_role"))
        spec = specs.get(key)
        if spec is None:
            print(f"  warn: no spec for {key} — skipping")
            continue

        # Reroute V4-pro reasoning from OpenRouter/Novita to DeepSeek-direct
        # for the retry. PREREGISTRATION amendment: removes third-party
        # serving variability, keeps both members of the V4-pro pair on the
        # same first-party API, and exploits the 75% promotional discount +
        # provider-side prompt caching. V4-flash is already on DeepSeek-direct.
        if spec.family == "deepseek-v4-pro":
            client = get_client(
                "deepseek_direct",
                "deepseek-v4-pro",
                reasoning_config={"enabled": True},
            )
            provider_used = "deepseek_direct"
            upstream_pinned = "deepseek_direct"
        else:
            client = get_client(
                spec.provider, spec.model,
                upstream_provider=spec.upstream_provider,
                reasoning_config=spec.reasoning_config,
            )
            provider_used = spec.provider
            upstream_pinned = spec.upstream_provider

        try:
            turns, inst = builder(task, row)
        except Exception as e:
            print(f"  warn: rebuild failed for task={row.get('task_id')} "
                  f"sweep={row.get('sweep_value')}: {type(e).__name__}: {e}")
            continue

        async def _one(spec=spec, client=client, turns=turns, row=row,
                       inst=inst, upstream_pinned=upstream_pinned):
            tj = await run_trajectory(
                client, turns,
                task_id=row["task_id"],
                task_name=row.get("task_name"),
                experiment=axis,
                correct_answer=row["correct_answer"],
                scorer=inst.scorer,
                upstream_pinned=upstream_pinned,
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

    print(f"  launching {len(coros)} retry trajectories at concurrency={concurrency}")
    results = await run_batch(coros, concurrency=concurrency)
    n_recovered = sum(1 for t in results if not t.excluded)
    n_still_failed = sum(1 for t in results if t.excluded)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"  [{axis}] recovered={n_recovered}  still_failed={n_still_failed}  cost≈${cost:.4f}")
    return n_recovered


async def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    axes = ["self_consistency", "sycophancy", "context_rot"] if args.axis == "all" else [args.axis]
    total_recovered = 0
    for ax in axes:
        n = await retry_axis(ax, concurrency=args.concurrency, dry_run=args.dry_run,
                              max_tokens=args.max_tokens)
        total_recovered += n
    print(f"\nTOTAL: {'planned' if args.dry_run else 'recovered'} {total_recovered} rows across {len(axes)} axes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["all", "self_consistency", "sycophancy", "context_rot"],
                   default="all")
    p.add_argument("--concurrency", type=int, default=6,
                   help="Lower concurrency = gentler on providers, esp. since "
                        "high max_tokens makes each call heavier.")
    p.add_argument("--max-tokens", type=int, default=NEW_MAX_TOKENS,
                   help="Per-call max_tokens (default 8192). Use 16384 or 32768 "
                        "for context_rot reasoning cells whose long input "
                        "conversations + reasoning_content blow the smaller budget.")
    p.add_argument("--dry-run", action="store_true",
                   help="Just count what would be retried, don't make API calls.")
    asyncio.run(main(p.parse_args()))
