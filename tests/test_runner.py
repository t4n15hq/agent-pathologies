import asyncio
import json
from pathlib import Path

import pytest

from agent_pathologies.client import MockClient
from agent_pathologies.runner import cell_key, existing_cell_keys, run_trajectory, write_jsonl
from agent_pathologies.tasks.arithmetic import MultiStepArithmetic
from agent_pathologies.types import ModelRole


def test_cell_key_is_deterministic():
    a = cell_key("m1", "task-1", {"x": 1}, 42, model_role="instruct")
    b = cell_key("m1", "task-1", {"x": 1}, 42, model_role="instruct")
    assert a == b


def test_cell_key_differs_on_any_input_change():
    base = cell_key("m1", "task-1", {"x": 1}, 42, model_role="instruct")
    assert cell_key("m2", "task-1", {"x": 1}, 42, model_role="instruct") != base
    assert cell_key("m1", "task-2", {"x": 1}, 42, model_role="instruct") != base
    assert cell_key("m1", "task-1", {"x": 2}, 42, model_role="instruct") != base
    assert cell_key("m1", "task-1", {"x": 1}, 43, model_role="instruct") != base


def test_cell_key_differs_by_model_role():
    """Same model+task+sweep+seed but different roles must NOT collide.
    Without this, V4-pro instruct and reasoning siblings (same model ID,
    only reasoning_config differs) would clobber each other on resume."""
    instr = cell_key("deepseek/deepseek-v4-pro", "arith-h5-0", 0, 42, model_role="instruct")
    reas = cell_key("deepseek/deepseek-v4-pro", "arith-h5-0", 0, 42, model_role="reasoning")
    assert instr != reas, "same-model siblings must have distinct cell keys"


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
    assert cell_key(tj.model, tj.task_id, tj.sweep_value, tj.seed,
                    model_role=tj.model_role) in keys


def test_excluded_rows_count_as_attempted(tmp_path: Path):
    """Excluded trajectories are reported, not silently backfilled on resume."""
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
        "model_role": "instruct",
        "extra": {"cell_key": cell_key("m1", "t-1", 0, 1, model_role="instruct")},
    }
    out.write_text(json.dumps(row) + "\n")
    assert existing_cell_keys(out) == {cell_key("m1", "t-1", 0, 1, model_role="instruct")}


def test_provider_empty_response_distinguished_from_empty_probe():
    """A trajectory whose provider returned no completion tokens at all
    must be classified as `provider_empty_response`, not `empty_probe_answer`.
    Discovered post-stage-1: DeepSeek reasoning sometimes returns 200 OK
    with empty body — bumping max_tokens won't fix it."""
    from agent_pathologies.analysis.exclusions import exclusion_reason
    from agent_pathologies.types import Trajectory

    empty_body = Trajectory(
        task_id="t", experiment="test", model="m", provider="p",
        probe_answer="", output_tokens=0,
    )
    assert exclusion_reason(empty_body) == "provider_empty_response"

    # Distinguish from empty_probe with non-zero tokens (e.g. all whitespace)
    only_whitespace = Trajectory(
        task_id="t", experiment="test", model="m", provider="p",
        probe_answer="   ", output_tokens=5,
    )
    assert exclusion_reason(only_whitespace) == "empty_probe_answer"


def test_reclassify_legacy_empty_probe_answer():
    """The legacy `empty_probe_answer` label gets retroactively rewritten
    to `provider_empty_response` for rows where output_tokens==0 + empty
    content, but only those rows."""
    import pandas as pd
    from agent_pathologies.analysis.metrics import reclassify_legacy_exclusions

    df = pd.DataFrame([
        {"exclusion_reason": "empty_probe_answer", "probe_answer": "",  "output_tokens": 0},
        {"exclusion_reason": "empty_probe_answer", "probe_answer": "  ", "output_tokens": 0},
        {"exclusion_reason": "empty_probe_answer", "probe_answer": "x", "output_tokens": 5},  # real empty
        {"exclusion_reason": "refusal_detected",   "probe_answer": "",  "output_tokens": 0},  # don't touch
    ])
    out = reclassify_legacy_exclusions(df)
    assert out.iloc[0]["exclusion_reason"] == "provider_empty_response"
    assert out.iloc[1]["exclusion_reason"] == "provider_empty_response"
    assert out.iloc[2]["exclusion_reason"] == "empty_probe_answer"
    assert out.iloc[3]["exclusion_reason"] == "refusal_detected"


def test_extracted_divergence_vs_string_divergence():
    """Per the analysis-agent finding: string divergence over-counts surface
    variation. Extracted-integer divergence collapses CoT-length variation
    that doesn't change the actual answer."""
    from agent_pathologies.analysis.metrics import (
        answer_divergence, extracted_divergence,
    )
    # Same integer 42, different CoT prefixes — string divergence sees 3
    # unique strings; integer divergence sees one consistent answer (42).
    answers = [
        "Let me compute step by step. The answer is 42.",
        "Step 1: ... Step 2: ... So the answer is 42.",
        "The answer is 42.",
    ]
    assert answer_divergence(answers) > 0.5     # high — they're all different strings
    assert extracted_divergence(answers) == 0.0  # zero — same integer


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
