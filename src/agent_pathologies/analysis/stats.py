"""Paired statistical tests for Pivot A.

The headline test is McNemar's exact (binomial) test for paired binary
outcomes — appropriate when the same task_seed is run on both members of
an instruct/reasoning pair. We use the exact form (not the asymptotic
chi-squared) because b+c can be small per cell.

Effect size is Cohen's h on paired proportions. Multiple-comparisons
correction is Benjamini-Hochberg across all (pair × cell) tests within an
axis, with FDR target = 0.05 (per PREREGISTRATION §4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import binomtest


@dataclass
class PairedResult:
    n_concordant_correct: int     # both right
    n_concordant_wrong: int       # both wrong
    b: int                        # only model A right (A correct, B wrong)
    c: int                        # only model B right
    p_value: float                # McNemar's exact two-sided
    p_a: float                    # marginal accuracy of A
    p_b: float                    # marginal accuracy of B
    cohens_h: float
    n: int                        # total paired observations


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test. b and c are the off-diagonal counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue)


def cohens_h(p1: float, p2: float) -> float:
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def paired_test(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> PairedResult:
    """Compare two paired binary sequences. a_correct[i] and b_correct[i]
    must be the same task instance under conditions A and B."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired sequences must have equal length")
    n = len(a_correct)
    if n == 0:
        return PairedResult(0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0)

    both_right = sum(1 for a, b in zip(a_correct, b_correct) if a and b)
    both_wrong = sum(1 for a, b in zip(a_correct, b_correct) if not a and not b)
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)

    p_val = mcnemar_exact(only_a, only_b)
    p_a = (both_right + only_a) / n
    p_b = (both_right + only_b) / n
    h = cohens_h(p_a, p_b)

    return PairedResult(
        n_concordant_correct=both_right,
        n_concordant_wrong=both_wrong,
        b=only_a,
        c=only_b,
        p_value=p_val,
        p_a=p_a,
        p_b=p_b,
        cohens_h=h,
        n=n,
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_iters, dtype=float)
    for i in range(n_iters):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = statistic(sample)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class DiDResult:
    """Paired difference-in-differences for sycophancy.

    `did = (reas_correct − reas_wrong) − (instr_correct − instr_wrong)`,
    computed over the shared task_id index. Positive `did` means reasoning
    training widens the responsiveness gap between honest and dishonest
    pushback (i.e., reasoning models are more sycophancy-resistant in the
    relative sense)."""
    did: float
    ci_lo: float
    ci_hi: float
    bootstrap_p: float           # two-sided
    instr_gap: float             # acc(correct) − acc(wrong) for instruct
    reas_gap: float
    n_paired: int


def paired_did_bootstrap(
    *,
    instr_correct: Sequence[bool],
    instr_wrong: Sequence[bool],
    reas_correct: Sequence[bool],
    reas_wrong: Sequence[bool],
    n_iters: int = 10_000,
    seed: int = 42,
) -> DiDResult:
    """All four sequences must be aligned by task_id index and the same
    length n. The bootstrap resamples shared task_ids with replacement and
    recomputes the four-cell mean each iteration, preserving the paired
    structure."""
    n = len(instr_correct)
    if not (n == len(instr_wrong) == len(reas_correct) == len(reas_wrong)):
        raise ValueError("all four arms must have equal length")
    if n == 0:
        return DiDResult(0.0, float("nan"), float("nan"), 1.0, 0.0, 0.0, 0)

    ic = np.asarray(instr_correct, dtype=float)
    iw = np.asarray(instr_wrong, dtype=float)
    rc = np.asarray(reas_correct, dtype=float)
    rw = np.asarray(reas_wrong, dtype=float)

    instr_gap = float(ic.mean() - iw.mean())
    reas_gap = float(rc.mean() - rw.mean())
    did = reas_gap - instr_gap

    rng = np.random.default_rng(seed)
    boots = np.empty(n_iters, dtype=float)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = (rc[idx].mean() - rw[idx].mean()) - (ic[idx].mean() - iw[idx].mean())

    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    # Two-sided bootstrap p: 2 × min(P(did_b > 0), P(did_b < 0)), capped at 1.
    p_pos = float((boots > 0).mean())
    p_neg = float((boots < 0).mean())
    bootstrap_p = min(1.0, 2.0 * min(p_pos, p_neg)) if (p_pos + p_neg) > 0 else 1.0

    return DiDResult(
        did=float(did),
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        bootstrap_p=bootstrap_p,
        instr_gap=instr_gap,
        reas_gap=reas_gap,
        n_paired=n,
    )


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Returns BH-adjusted q-values in the same order as input p-values."""
    if not pvalues:
        return []
    try:
        from statsmodels.stats.multitest import multipletests  # type: ignore
        _, qvals, _, _ = multipletests(list(pvalues), alpha=alpha, method="fdr_bh")
        return [float(q) for q in qvals]
    except ImportError:
        # Manual BH if statsmodels is unavailable
        n = len(pvalues)
        order = sorted(range(n), key=lambda i: pvalues[i])
        adj = [0.0] * n
        prev = 1.0
        for rank, idx in enumerate(reversed(order)):
            i = n - rank
            q = pvalues[idx] * n / i
            prev = min(prev, q)
            adj[idx] = min(prev, 1.0)
        return adj
