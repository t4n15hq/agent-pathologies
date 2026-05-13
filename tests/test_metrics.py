from agent_pathologies.analysis.metrics import accuracy_with_ci, answer_divergence


def test_divergence_zero_when_all_same():
    assert answer_divergence(["yes", "yes", "yes"]) == 0.0


def test_divergence_above_zero_with_disagreement():
    # 4 same, 1 different → mode_freq=4, n=5, divergence = 1/5
    d = answer_divergence(["yes", "yes", "yes", "yes", "no"])
    assert abs(d - 0.2) < 1e-9


def test_divergence_ignores_whitespace_case():
    assert answer_divergence(["yes", "YES ", " Yes"]) == 0.0


def test_divergence_empty_input_returns_zero():
    assert answer_divergence([]) == 0.0
    assert answer_divergence([None, "", "   "]) == 0.0


def test_accuracy_with_ci_basic():
    stats = accuracy_with_ci([True] * 8 + [False] * 2, n_iters=1000)
    assert stats["n"] == 10
    assert abs(stats["accuracy"] - 0.8) < 1e-9
    assert 0.0 <= stats["ci_lo"] <= stats["accuracy"] <= stats["ci_hi"] <= 1.0


def test_accuracy_with_ci_empty_returns_nan():
    stats = accuracy_with_ci([], n_iters=100)
    assert stats["n"] == 0
    # NaN equality check
    assert stats["accuracy"] != stats["accuracy"]
