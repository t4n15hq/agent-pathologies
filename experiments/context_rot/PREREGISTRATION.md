# Pre-Registration — context_rot

Locked alongside the root [PREREGISTRATION.md](../../PREREGISTRATION.md).

## H₀ / H₁

- **H₀:** Within a model family, instruct and reasoning siblings show the
  same accuracy on a running state-tracking task across the
  (n_filler × kind) sweep.
- **H₁:** They differ — reasoning training shifts the slope (or intercept)
  of the decay curve.

Two-sided.

## Operationalization

- Task: `VariableTracking` (n_updates = 5). The model must maintain a running
  integer state across several earlier update turns, then report the final
  value at probe. This gives more headroom than single-fact recall and is
  scored with strict integer extraction.
- Sweep:
  - `n_filler ∈ {0, 2, 5, 10, 20, 40}` user/assistant pairs inserted between
    the state-update setup and the final probe.
  - `kind ∈ {irrelevant, related, token_matched, collapsed}`.
- Pairing: same `task_seed` × same `(kind, n_filler)` cell on both instruct
  and reasoning members of each family.

## Control conditions

- `irrelevant` filler — natural chitchat unrelated to the running variable.
- `related` filler — math-adjacent but not about the tracked variable. Tests
  whether topical proximity worsens state retention.
- `token_matched` filler — pre-filled short turns with fixed transcript length.
  Controls generated/freeform filler content **at the same turn count**.
- `collapsed` filler — **the true turn-vs-token control.** A single
  user+assistant pair (2 turns) whose total token mass equals
  `n_filler × AVG_TOKENS_PER_IRRELEVANT_PAIR`. Comparing accuracy at
  (kind=irrelevant, k=20) vs (kind=collapsed, k=20) holds total token mass
  approximately constant while changing turn count from 40 to 2 — this
  isolates the *turn count* dimension of context rot from the *token mass*
  dimension. (See `src/agent_pathologies/conversation/synthesizer.py:
  build_filler_block` and `tests/test_synthesizer.py`.)

## Amendment log

- **2026-05-13:** Added `collapsed` filler kind so the sweep includes a true
  turn-vs-token control (previously the disclaimer-only state). The kind
  produces a single fat user+assistant pair sized to match the cumulative
  token mass of k small filler pairs.

## Statistical test

- **Paired McNemar's exact** on `is_correct` per (family × kind × n_filler) cell.
- **Cohen's h** for paired-proportion effect size.
- BH correction across all (family × kind × n_filler) cells within this
  experiment.

## Sample size

N = 50 task instances × 6 filler counts × 4 kinds × (instruct + reasoning) per
family = 2,400 trajectories per family. This gives ≥ 80% power to detect
h = 0.25 at α = 0.05 per cell.

## Falsification

For each family: if max |h| across all cells < 0.20 *and* min BH-corrected q
across all cells > 0.05, reasoning training is reported as having no
detectable effect on context-rot behavior in that family. We report this
null result — we do not chase a different perturbation type.
