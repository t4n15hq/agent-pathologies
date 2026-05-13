import asyncio
import json
from pathlib import Path

import pytest

from agent_pathologies.client import MockClient
from agent_pathologies.runner import cell_key, existing_cell_keys, run_trajectory, write_jsonl
from agent_pathologies.tasks.arithmetic import MultiStepArithmetic
from agent_pathologies.types import ModelRole


def test_cell_key_is_deterministic():
    a = cell_key("m1", "task-1", {"x": 1}, 42)
    b = cell_key("m1", "task-1", {"x": 1}, 42)
    assert a == b


def test_cell_key_differs_on_any_input_change():
    base = cell_key("m1", "task-1", {"x": 1}, 42)
    assert cell_key("m2", "task-1", {"x": 1}, 42) != base
    assert cell_key("m1", "task-2", {"x": 1}, 42) != base
    assert cell_key("m1", "task-1", {"x": 2}, 42) != base
    assert cell_key("m1", "task-1", {"x": 1}, 43) != base


def test_existing_cell_keys_resumability(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    # Empty file -> empty set
    assert existing_cell_keys(out) == set()

    async def go():
        client = MockClient()
        task = MultiStepArithmetic()
        inst = task.sample(0)
        tj = await run_trajectory(
            client,
            inst.setup_turns,
            task_id=inst.task_id,
            task_name=inst.task_name,
            experiment="test",
            correct_answer=inst.correct_answer,
            scorer=inst.scorer,
            seed=0,
            sweep_value=0,
            model_role=ModelRole.INSTRUCT,
            model_family="mockfam",
        )
        write_jsonl(tj, out)
        return tj

    tj = asyncio.run(go())
    assert tj.is_correct is True

    keys = existing_cell_keys(out)
    assert len(keys) == 1
    # The same key recomputed independently is in the set.
    assert cell_key(tj.model, tj.task_id, tj.sweep_value, tj.seed) in keys


def test_excluded_rows_not_counted_as_done(tmp_path: Path):
    """If a trajectory is excluded, resuming should re-attempt it."""
    out = tmp_path / "log.jsonl"
    # Hand-craft an excluded row
    row = {
        "trajectory_id": "x",
        "task_id": "t-1",
        "experiment": "test",
        "model": "m1",
        "provider": "mock",
        "sweep_value": 0,
        "seed": 1,
        "excluded": True,
        "exclusion_reason": "refusal_detected",
        "extra": {"cell_key": cell_key("m1", "t-1", 0, 1)},
    }
    out.write_text(json.dumps(row) + "\n")
    assert existing_cell_keys(out) == set()


def test_runner_attaches_model_family_role_and_tokens():
    async def go():
        client = MockClient()
        task = MultiStepArithmetic()
        inst = task.sample(0)
        return await run_trajectory(
            client,
            inst.setup_turns,
            task_id=inst.task_id,
            task_name=inst.task_name,
            experiment="test",
            correct_answer=inst.correct_answer,
            scorer=inst.scorer,
            seed=0,
            sweep_value=0,
            model_role=ModelRole.REASONING,
            model_family="mockfam",
        )

    tj = asyncio.run(go())
    assert tj.model_family == "mockfam"
    assert tj.model_role == ModelRole.REASONING
    assert tj.input_tokens and tj.input_tokens > 0
    assert tj.output_tokens and tj.output_tokens > 0
    assert tj.excluded is False
