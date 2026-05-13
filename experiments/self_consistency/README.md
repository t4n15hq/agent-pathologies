# Self-Consistency Drift

## Hypothesis

At `temperature=0` with a fixed seed, an agent's answer to the same prompt across
N replays should be identical. Quantify the actual divergence — this is the
**noise floor** for `context_rot` and `sycophancy`.

## Why this matters

Without a measured noise floor, the effect sizes in the other two experiments
are uninterpretable. If divergence is high here, the perturbation experiments
need larger sample sizes to detect real effects.

## Protocol

- Sample `T` task instances (default 20) of `MultiStepArithmetic`.
- For each, run `R` independent replays (default 20).
- `temperature=0`, seed varies across replays only.
- Measure:
  - **Divergence:** `|unique(answers)| / R` per task. 0 = perfect consistency.
  - **Accuracy:** sanity check; correctness should be roughly constant.

## What success looks like

- Either: divergence is near zero across providers (deterministic-as-advertised);
- Or: divergence is nonzero, with a clear breakdown by provider — that itself
  is a publishable observation, given how often "temperature 0 = deterministic"
  is assumed in agent papers.

## Run

```bash
# Smoke test
python experiments/self_consistency/run.py
python experiments/self_consistency/analyze.py

# Real run, Qwen-72B via Together
python experiments/self_consistency/run.py \
    --provider together \
    --model "Qwen/Qwen2.5-72B-Instruct-Turbo" \
    --n-tasks 50 --n-repeats 30
```
