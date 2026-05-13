# Pre-Registration — Pivot A

**Locked:** 2026-05-13. **Author:** Tanishq (t4n15hq). **Frozen sections:** §1–§6
below. Any deviation must be documented as an amendment with a date and reason.

The purpose of this document is to commit to the hypothesis, model set,
analysis plan, and stopping rules *before* any data is collected, so the
paper cannot be retroactively shaped to fit the data (HARKing).

---

## §1 Research question

> Do reasoning-tuned LLMs exhibit different multi-turn trajectory pathologies
> than their instruct-tuned siblings, when scale and base architecture are
> held constant?

Three pathology axes are measured: self-consistency drift, context rot,
sycophancy persistence. Each is operationalized in `experiments/<axis>/`.

## §2 Pairs under test

Within-family instruct vs reasoning pairs (see `configs/models.yaml`):

1. `deepseek-v4-pro`  ↔  `deepseek-r1-0528`
2. `qwen3-235b-a22b-2507`  ↔  `qwen3-235b-a22b-thinking-2507`
3. `qwen3-30b-a3b-instruct-2507`  ↔  `qwen3-30b-a3b-thinking-2507`

Closed-frontier anchor (no pair): `anthropic/claude-opus-4.7` (instruct-style).
The anchor is **not** part of the paired test; it's reported as a single column
for context only.

## §3 Hypotheses

For each axis × each pair we test a directional pair:

| Axis | H₀ | H₁ (the finding we'd publish either way) |
|---|---|---|
| self_consistency | No within-pair difference in answer divergence | reasoning > instruct divergence (or <) |
| context_rot | No within-pair difference in accuracy decay slope vs. n_filler | reasoning more (or less) robust to filler |
| sycophancy | No within-pair difference in accuracy at re-probe after wrong pushback | reasoning more (or less) resistant |

The directionality is intentionally two-sided: we expect either direction is
interesting and we don't bias the protocol toward one.

## §4 Primary metric and effect-size threshold

- **Primary metric:** `is_correct` at the designated probe turn (binary).
- **Statistical test:** McNemar's exact test for paired binary outcomes,
  computed per (pair × sweep-cell) and pooled across task instances.
- **Effect-size threshold:** **Cohen's h ≥ 0.20** on paired proportions. Below
  this we call the result a null finding *even if p < 0.05* (publication of
  null is fine; we'll report it).
- **Multiple-comparisons correction:** Benjamini-Hochberg across all pair ×
  cell tests within an axis, with FDR target = 0.05.

## §5 Sample size and stopping rule

Sample sizes are set in `configs/pivot_a.yaml` (n_tasks per axis). They are
chosen to give at least 80% power to detect h = 0.25 at α = 0.05, assuming a
correlation of 0.4 between paired outcomes. **No optional stopping**: we run
the full N and only then analyze.

## §6 Exclusions and quality controls

A trajectory is excluded from analysis if **and only if**:

1. The provider returned an HTTP error after retries (logged in `extra.error`).
2. The model refused to answer (e.g., safety refusal) — detected by absence of
   any digit in the probe answer for arithmetic tasks, or absence of any
   plausible value for needle tasks. Detection rule is in
   `src/agent_pathologies/analysis/exclusions.py` and frozen with this doc.
3. Output exceeded `max_tokens` (truncated). Detected by missing terminating
   punctuation **and** length == max_tokens.

Excluded trajectories are reported as a separate count per cell. We will
**not** silently re-sample to backfill exclusions.

## §7 Reporting (non-frozen)

The final paper will report, per axis:
- A paired bar chart per family (instruct vs reasoning) with bootstrap 95% CIs.
- A table of McNemar p-values and Cohen's h per (pair × cell), with BH-adjusted
  q-values.
- The anchor model's bare numbers in a separate row, not in the paired test.
- All exclusions with reasons.
- All raw JSONL trajectories will be released alongside the paper.

## §8 What would falsify the headline?

If McNemar across the three pairs returns q > 0.05 for **all** three axes and
|h| < 0.20 for **all** pairs × cells, we conclude that reasoning training does
**not** systematically alter multi-turn pathology resistance. This is itself a
publishable null result and **the paper does not get reframed** post-hoc into
"a survey of pathologies on Chinese open-weight models" — that would be an
amendment, dated, and require a fresh sweep.

---

## Amendment log

*(none yet)*
