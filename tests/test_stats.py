import math

from agent_pathologies.analysis.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_h,
    mcnemar_exact,
    paired_test,
)


def test_mcnemar_zero_off_diagonal_pvalue_one():
    # When b = c = 0, no information → p = 1.
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric_off_diagonal_pvalue_one():
    # b == c, perfectly symmetric disagreement → p = 1.
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_extreme_off_diagonal_small_pvalue():
    # 10 vs 0 — heavily one-sided → p should be small.
    p = mcnemar_exact(10, 0)
    assert p < 0.01


def test_cohens_h_zero_when_proportions_equal():
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(0.0, 0.0) == 0.0


def test_cohens_h_increases_with_gap():
    h_small = cohens_h(0.50, 0.55)
    h_med = cohens_h(0.50, 0.65)
    h_large = cohens_h(0.50, 0.80)
    assert h_small < h_med < h_large


def test_paired_test_returns_expected_counts():
    # 10 tasks: 6 both right, 1 both wrong, 2 only A, 1 only B
    a = [True] * 6 + [False] * 1 + [True] * 2 + [False] * 1
    b = [True] * 6 + [False] * 1 + [False] * 2 + [True] * 1
    r = paired_test(a, b)
    assert r.n_concordant_correct == 6
    assert r.n_concordant_wrong == 1
    assert r.b == 2
    assert r.c == 1
    assert r.n == 10


def test_bootstrap_ci_brackets_known_mean():
    values = [1.0] * 50 + [0.0] * 50
    lo, hi = bootstrap_ci(values, n_iters=2000, seed=1)
    assert lo < 0.5 < hi


def test_benjamini_hochberg_monotone_and_bounded():
    ps = [0.001, 0.01, 0.03, 0.2, 0.5]
    qs = benjamini_hochberg(ps)
    assert len(qs) == len(ps)
    # Largest p maps to largest q
    assert qs[ps.index(max(ps))] == max(qs)
    # All q ≤ 1
    assert all(0 <= q <= 1 for q in qs)
    # q ≥ p for each (BH never *decreases* the p)
    for p, q in zip(ps, qs):
        assert q + 1e-9 >= p


def test_benjamini_hochberg_empty_list():
    assert benjamini_hochberg([]) == []
