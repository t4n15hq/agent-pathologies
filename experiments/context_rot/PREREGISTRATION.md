# Pre-Registration — context_rot

Locked alongside the root [PREREGISTRATION.md](../../PREREGISTRATION.md).

## H₀ / H₁

- **H₀:** Within a model family, instruct and reasoning siblings show the
  same accuracy on a planted fact across the (n_filler × kind) sweep.
- **H₁:** They differ — reasoning training shifts the slope (or intercept)
  of the decay curve.

Two-sided.

## Operationalization

- Task: `MultiFactNeedle` (n_facts = 4). Multiple facts planted simultaneously;
  one is probed. Strict word-boundary scoring on the planted value.
- Sweep:
  - `n_filler ∈ {0, 2, 5, 10, 20, 40}` user/assistant pairs inserted between
    plant and probe.
  - `kind ∈ {irrelevant, related, token_matched}`.
- Pairing: same `task_seed` × same `(kind, n_filler)` cell on both instruct
  and reasoning members of each family.

## Control conditions

- `irrelevant` filler — natural chitchat unrelated to the planted facts.
- `related` filler — on-topic but not about the planted facts. Tests
  whether topical proximity worsens forgetting.
- `token_matched` filler — pre-filled short turns with controlled token
  count. **This is the critical control:** comparing accuracy at
  (kind=irrelevant, k=20) vs (kind=token_matched, k=20) isolates the
  effect of *turn structure* from the confound of *token count*. If
  accuracy is the same, what matters is just the number of turns. If
  accuracy differs, content type or token volume drives the effect.

## Statistical test

- **Paired McNemar's exact** on `is_correct` per (family × kind × n_filler) cell.
- **Cohen's h** for paired-proportion effect size.
- BH correction across all (family × kind × n_filler) cells within this
  experiment.

## Sample size

N = 50 task instances × 6 filler counts × 3 kinds × (instruct + reasoning) per
family = 1,800 trajectories per family. This gives ≥ 80% power to detect
h = 0.25 at α = 0.05 per cell.

## Falsification

For each family: if max |h| across all cells < 0.20 *and* min BH-corrected q
across all cells > 0.05, reasoning training is reported as having no
detectable effect on context-rot behavior in that family. We report this
null result — we do not chase a different perturbation type.
