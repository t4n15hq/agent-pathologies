"""Dry-run cost estimator. Reads configs/, simulates trajectory lengths
without calling any model, and reports expected $ per model and total."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agent_pathologies.budget import CostSpec, count_messages, count_tokens
from agent_pathologies.config_loader import iter_run_specs, load_yaml
from agent_pathologies.conversation.synthesizer import filler_turn_pair
from agent_pathologies.conversation.pushback import pushback as make_pushback
from agent_pathologies.tasks import get_task
from agent_pathologies.types import Role, Turn


def _simulate_trajectory_tokens(turns: list[Turn], avg_response_tokens: int = 25
                                ) -> tuple[int, int]:
    """Approximate input/output tokens a trajectory of these turns would use."""
    input_tokens = 0
    output_tokens = 0
    history: list[dict[str, str]] = []
    for t in turns:
        if t.role == Role.ASSISTANT and not t.content:
            input_tokens += count_messages(history)
            output_tokens += avg_response_tokens
            history.append({"role": "assistant", "content": "x" * (avg_response_tokens * 4)})
        else:
            history.append({"role": t.role.value, "content": t.content})
    return input_tokens, output_tokens


def estimate_self_consistency(cfg: dict, spec_cost: CostSpec) -> tuple[int, int, int, float]:
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    inst = task.sample(0)
    n_traj = cfg["n_tasks"] * cfg["n_repeats"]
    inp, out = _simulate_trajectory_tokens(list(inst.setup_turns), avg_response_tokens=20)
    return n_traj, inp, out, n_traj * spec_cost.for_call(inp, out)


def estimate_context_rot(cfg: dict, spec_cost: CostSpec) -> tuple[int, int, int, float]:
    import random
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    inst = task.sample(0)
    total_calls = 0
    total_inp = 0
    total_out = 0
    total_cost = 0.0
    rng = random.Random(0)
    for kind in cfg["filler_kinds"]:
        for k in cfg["filler_counts"]:
            turns = list(inst.setup_turns)
            for _ in range(k):
                turns.extend(filler_turn_pair(kind, rng))
            turns.append(Turn(role=Role.USER, content=inst.probe_question))
            turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
            inp, out = _simulate_trajectory_tokens(turns, avg_response_tokens=15)
            cost = spec_cost.for_call(inp, out)
            n_for_cell = cfg["n_tasks"]
            total_calls += n_for_cell
            total_inp += n_for_cell * inp
            total_out += n_for_cell * out
            total_cost += n_for_cell * cost
    return total_calls, total_inp // max(total_calls, 1), total_out // max(total_calls, 1), total_cost


def estimate_sycophancy(cfg: dict, spec_cost: CostSpec) -> tuple[int, int, int, float]:
    import random
    task = get_task(cfg["task"], **cfg.get("task_kwargs", {}))
    inst = task.sample(0)
    total_calls = 0
    total_inp = 0
    total_out = 0
    total_cost = 0.0
    rng = random.Random(0)
    for cond in cfg["pushback_conditions"]:
        for gap in cfg["post_pushback_gaps"]:
            turns = list(inst.setup_turns)
            turns.extend(make_pushback(inst.correct_answer, cond))
            for _ in range(gap):
                turns.extend(filler_turn_pair("irrelevant", rng))
            turns.append(Turn(role=Role.USER, content=f"Just to confirm: {inst.probe_question}"))
            turns.append(Turn(role=Role.ASSISTANT, content="", is_probe=True))
            inp, out = _simulate_trajectory_tokens(turns, avg_response_tokens=20)
            cost = spec_cost.for_call(inp, out)
            n_for_cell = cfg["n_tasks"]
            total_calls += n_for_cell
            total_inp += n_for_cell * inp
            total_out += n_for_cell * out
            total_cost += n_for_cell * cost
    return total_calls, total_inp // max(total_calls, 1), total_out // max(total_calls, 1), total_cost


def main(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.config))
    models_cfg = load_yaml(Path(args.models_config))
    specs = list(iter_run_specs(models_cfg))

    print(f"{'model':<55s}  {'exp':<18s}  {'#traj':>8s}  {'avgIn':>7s}  {'avgOut':>7s}  {'$':>8s}")
    print("-" * 110)
    grand = 0.0
    for spec in specs:
        for exp_name, est_fn, sub in [
            ("self_consistency", estimate_self_consistency, cfg["self_consistency"]),
            ("context_rot",      estimate_context_rot,      cfg["context_rot"]),
            ("sycophancy",       estimate_sycophancy,       cfg["sycophancy"]),
        ]:
            n_traj, inp, out, cost = est_fn(sub, spec.cost_spec)
            grand += cost
            print(f"{spec.model:<55s}  {exp_name:<18s}  {n_traj:>8d}  {inp:>7d}  {out:>7d}  ${cost:>7.2f}")
    print("-" * 110)
    print(f"GRAND TOTAL ESTIMATE: ${grand:.2f}")
    print("(Estimate assumes short responses. Reasoning models with long")
    print(" thinking traces can exceed this 3-10x — budget headroom.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pivot_a.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    main(p.parse_args())
