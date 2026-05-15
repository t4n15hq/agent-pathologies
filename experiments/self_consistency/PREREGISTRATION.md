# Pre-Registration — self_consistency

Locked alongside the root [PREREGISTRATION.md](../../PREREGISTRATION.md).

## H₀ / H₁

- **H₀:** Within a model family, instruct and reasoning siblings have the
  same answer-divergence distribution across replays of the same task at T=0.
- **H₁:** Within a model family, instruct and reasoning siblings have
  *different* answer-divergence distributions.

We are two-sided. We have no strong prior on which direction reasoning
training pushes divergence (reasoning models could be more consistent because
they re-derive, or *less* consistent because their longer chains amplify any
sampling noise).

## Operationalization

- Task: `MultiStepArithmetic` (hardness=3) — strict integer scoring.
- Per-task divergence: fraction of N replays whose extracted answer differs
  from the modal answer.
- Per-task mean is paired across roles by `task_id` — same `task_seed` is
  run on both instruct and reasoning members of the pair.
- Request payloads are identical across replays for a given `(model, task_id)`:
  temperature is 0 and the API seed is held fixed at `task_seed`. This makes
  the experiment a true deterministic-settings replay test rather than a
  seed-sensitivity test.

## Statistical test

- **Paired Wilcoxon signed-rank** on divergence values per family.
  Wilcoxon is appropriate because (a) divergence per task is bounded and
  not Gaussian, (b) zero-differences are common (perfect consistency).
- Bootstrap 95% CI on the per-task `(instruct - reasoning)` divergence
  difference.
- BH-corrected q-values across families.

## Stopping rule

Run the full N = 40 tasks × 25 replays per model. No optional
stopping.

## Exclusions

Per root §6. Excluded trajectories are dropped from divergence computation
*for that (family, task)* but the task is still included if ≥ 3 replays
survive on each side.

## Falsification

If q ≥ 0.05 for **all** families *and* the per-task delta has |median| <
0.05 (5 percentage points of divergence) for all families, we report a
null finding for this axis. Self-consistency is reported either way because
it sets the noise floor for the other two axes.

## Amendment log

- **2026-05-14:** Two changes after stage-1 calibration data revealed
  metric issues:
  1. **Co-primary accuracy-paired Wilcoxon added.** The original
     divergence-based Wilcoxon misses a real failure mode revealed at
     hardness=5: a model can be highly *consistent* in giving the *same
     wrong answer* (Qwen instruct emits "The answer is 1234." to ~65% of
     hardness-5 prompts). Divergence near zero but accuracy near zero too.
     Adding paired Wilcoxon on per-task accuracy captures this; we report
     both, BH-correcting within each metric family. A family is reported
     as positive only if **either** test crosses q < 0.05 with delta in
     the prereg direction.
  2. **Divergence reported on the extracted scored quantity, not the raw
     response string.** Chain-of-thought text varies in length even when
     the final integer is identical (the `deepseek-v4-flash instruct`
     stage-1 calibration data showed string-divergence = 0.88 vs
     integer-divergence = 0.42 on the same trajectories). The reported
     statistic is now `extracted_divergence` (using `extract_last_integer`
     on each probe answer) which isolates "did the model commit to the
     same answer" from CoT verbosity. String-divergence is kept in the CSV
     as a secondary column for transparency.
