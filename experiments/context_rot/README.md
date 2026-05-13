# Context Rot

## Hypothesis

As the number of intervening turns between state updates and the final probe
grows, accuracy on the probe degrades — even though all updates remain
in-context. Effect size depends on filler **kind**.

## Why this matters

The "lost in the middle" literature concerns *positional* effects in a single
prompt. This experiment measures the *temporal* analogue: long multi-turn
conversations degrading the model's grip on earlier state. Every agent builder
hits this; almost no one has measured it cleanly.

## Protocol

- Initialize and update a running variable across multiple setup turns
  (`VariableTracking`).
- Insert `k` user+assistant filler pairs, where `k ∈ {0, 2, 5, 10, 20, 40}`.
- Filler kinds: `irrelevant`, `related`, `token_matched`.
- Probe: ask for the current value of the variable.
- Score: strict integer extraction.

## What success looks like

- Monotonic decay curve, effect size larger than the noise floor from
  `self_consistency`.
- Cross-model: does the curve differ between instruct and reasoning siblings
  of the same family?
- Mechanistic (open-weights only): probe attention from the final query back
  to the state-update turns; correlate decay of attention with decay of
  accuracy. **This is the high-impact follow-up — get baseline curve first.**

## Run

```bash
python experiments/context_rot/run.py
python experiments/context_rot/analyze.py
```
