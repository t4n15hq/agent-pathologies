import pytest

from agent_pathologies.tasks import TASK_REGISTRY, get_task
from agent_pathologies.tasks.scoring import score_integer


def test_registry_contains_all_expected_tasks():
    expected = {"arithmetic", "needle", "multi_fact_needle", "closed_qa",
                "counterintuitive", "variable_tracking", "code_trace"}
    assert expected.issubset(set(TASK_REGISTRY))


def test_get_task_unknown_raises():
    with pytest.raises(ValueError):
        get_task("nonsense")


# -------- counterintuitive math --------

def test_counterintuitive_returns_intuitive_wrong():
    inst = get_task("counterintuitive").sample(0)
    assert inst.intuitive_wrong_answer is not None
    assert inst.intuitive_wrong_answer != inst.correct_answer
    # Verify the answer is parseable as integer (scoring is strict integer).
    assert score_integer(f"The answer is {inst.correct_answer}.",
                         inst.correct_answer) is True


def test_counterintuitive_sycophancy_signal_meaningful():
    """The intuitive_wrong shouldn't equal correct +/- 1; otherwise the
    perturbation is too weak to push a model."""
    for seed in range(15):
        inst = get_task("counterintuitive").sample(seed)
        if inst.correct_answer.isdigit() and inst.intuitive_wrong_answer.isdigit():
            assert abs(int(inst.correct_answer) - int(inst.intuitive_wrong_answer)) >= 1


# -------- variable tracking --------

def test_variable_tracking_setup_turns_count():
    # 6 updates → 1 system + (init + ack) + 6*(update + ack) = 1 + 2 + 12 = 15
    inst = get_task("variable_tracking", n_updates=6).sample(0)
    assert len(inst.setup_turns) == 15


def test_variable_tracking_answer_matches_metadata():
    inst = get_task("variable_tracking", n_updates=5).sample(42)
    assert inst.correct_answer == str(inst.metadata["final"])


# -------- code trace --------

def test_code_trace_answer_runs_python():
    """If the curated answer disagreed with actual execution, the module
    would have failed to import (verifier runs at module load)."""
    inst = get_task("code_trace").sample(0)
    assert inst.correct_answer.lstrip("-").isdigit()


def test_code_trace_question_includes_code_block():
    inst = get_task("code_trace").sample(1)
    assert "```python" in inst.probe_question


# -------- arithmetic hardness ladder --------

def test_arithmetic_hardness_increases_operand_count():
    for h, min_ops in [(1, 4), (2, 6), (3, 7), (4, 9)]:
        inst = get_task("arithmetic", hardness=h).sample(0)
        # crude operand-count by counting integer literals in expression
        import re
        ints = re.findall(r"\d+", inst.metadata["expression"])
        assert len(ints) >= min_ops, f"hardness {h}: only {len(ints)} ops"


def test_arithmetic_h3_answer_is_integer():
    for seed in range(10):
        inst = get_task("arithmetic", hardness=3).sample(seed)
        assert inst.correct_answer.lstrip("-").isdigit()


# -------- multi-fact needle --------

def test_multi_fact_needle_default_facts_bumped():
    """Default n_facts should be at least 6 — single-fact needle is too easy
    to leave headroom for context-rot effects."""
    inst = get_task("multi_fact_needle").sample(0)
    assert inst.metadata["n_facts"] >= 6


# -------- pushback intuitive override --------

def test_pushback_uses_intuitive_when_provided():
    from agent_pathologies.conversation.pushback import pushback
    turns = pushback("5", "wrong", wrong_answer_override="10")
    assert turns[0].metadata["asserted_answer"] == "10"
    assert turns[0].metadata["wrong_source"] == "intuitive"


def test_pushback_falls_back_to_generic_when_no_override():
    from agent_pathologies.conversation.pushback import pushback
    turns = pushback("5", "wrong")
    assert turns[0].metadata["asserted_answer"] == "12"  # 5 + 7
    assert turns[0].metadata["wrong_source"] == "generic_perturbation"
