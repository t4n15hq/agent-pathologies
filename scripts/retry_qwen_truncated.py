"""Retry Qwen instruct cells excluded as `truncated_at_max_tokens` with
raised `max_tokens=8192`.

Same root cause as the DeepSeek empties bug, just on a different model:
Qwen3-235B/-30B instruct on hardness-5 arithmetic genuinely needs >2048
output tokens to show work AND emit a final answer. Under our original
max_tokens=2048 the runner recorded these as `truncated_at_max_tokens`
exclusions. Empirically (median output_tokens at truncation = 2863), the
8K cap clears the ceiling for these prompts.

Only Qwen INSTRUCT cells are retried — Qwen reasoning cells didn't trip
truncation because we'd already been at max_tokens=2048 for them and
they hit `empty_probe_answer` instead (a different exclusion class).
Confined to sycophancy axis (the only place truncated_at_max_tokens
appears at scale; 38 cells across Qwen-235B-instruct and Qwen-30B-instruct).

Usage:
    python scripts/retry_qwen_truncated.py --dry-run    # plan only
    python scripts/retry_qwen_truncated.py              # execute
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
from agent_pathologies.conversation.synthesizer import filler_turn_pair
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
    """Match Qwen-instruct sycophancy cells that hit max_tokens truncation."""
    fam = row.get("model_family") or ""
    role = row.get("model_role") or ""
    reason = row.get("exclusion_reason") or ""
    return (
        row.get("excluded") is True
        and fam.startswith("qwen3")
        and role == "instruct"
        and reason == "truncated_at_max_tokens"
    )


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


async def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_yaml(Path("configs/pivot_a.yaml"))["sycophancy"]
    models_cfg = load_yaml(Path("configs/models.yaml"))
    specs = {(s.family, s.role.value): s for s in iter_run_specs(
        models_cfg, include_anchors=False)}
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))

    path = Path("data/sycophancy.jsonl")
    failed = [json.loads(l) for l in path.open() if l.strip() and is_target(json.loads(l))]
    print(f"qwen-instruct truncated_at_max_tokens cells eligible: {len(failed)}")

    if not failed:
        return

    from collections import Counter
    by_cell = Counter((r.get("model_family"), r.get("model_role")) for r in failed)
    for k, v in sorted(by_cell.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        print(f"\ndry-run: would re-attempt {len(failed)} cells at max_tokens={NEW_MAX_TOKENS}")
        return

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
            turns, inst = build_turns_sycophancy(task, row)
        except Exception as e:
            print(f"  warn: rebuild failed for {row.get('task_id')}: {type(e).__name__}: {e}")
            continue

        async def _one(spec=spec, client=client, turns=turns, row=row, inst=inst):
            tj = await run_trajectory(
                client, turns,
                task_id=row["task_id"],
                task_name=row.get("task_name"),
                experiment="sycophancy",
                correct_answer=row["correct_answer"],
                scorer=inst.scorer,
                upstream_pinned=spec.upstream_provider,
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

    print(f"\nlaunching {len(coros)} retries at concurrency={args.concurrency}")
    results = await run_batch(coros, concurrency=args.concurrency)
    recovered = sum(1 for t in results if not t.excluded)
    still_failed = sum(1 for t in results if t.excluded)
    cost = sum((t.cost_usd or 0.0) for t in results)
    print(f"recovered={recovered}  still_failed={still_failed}  cost≈${cost:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main(p.parse_args()))
