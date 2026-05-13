# Context Rot

## Hypothesis

As the number of intervening turns between a fact plant and a probe grows,
accuracy on the probe degrades — even when the fact remains in-context and
the model has the budget to attend to it. Effect size depends on filler
**kind** (irrelevant vs. related vs. contradictory).

## Why this matters

The "lost in the middle" literature concerns *positional* effects in a single
prompt. This experiment measures the *temporal* analogue: long multi-turn
conversations degrading the model's grip on early facts. Every agent builder
hits this; almost no one has measured it cleanly.

## Protocol

- Plant a `key = value` fact at turn 1 (`NeedleQA`).
- Insert `k` user+assistant filler pairs, where `k ∈ {0, 2, 5, 10, 20, 40}`.
- Filler kinds: `irrelevant`, `related`. (Contradictory is a TODO follow-up.)
- Probe: ask for the planted fact.
- Score: substring match against ground truth.

## What success looks like

- Monotonic decay curve, effect size larger than the noise floor from
  `self_consistency`.
- Cross-model: does the curve differ between Qwen-72B / DeepSeek-V3 / GLM-4.5?
  Different RLHF traditions would predict yes.
- Mechanistic (open-weights only): probe attention from the probe token
  back to the plant token; correlate decay of attention with decay of
  accuracy. **This is the high-impact follow-up — get baseline curve first.**

## Run

```bash
python experiments/context_rot/run.py
python experiments/context_rot/analyze.py
```
