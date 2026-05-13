# agent-pathologies

A controlled, multi-model framework for measuring **trajectory pathologies**
in multi-turn LLM agents — specifically comparing reasoning-tuned models
against their instruct-tuned siblings within each family (Pivot A; see
`RELATED_WORK.md` for prior-work analysis and `PREREGISTRATION.md` for the
locked hypotheses).

| Experiment | Hypothesis (paired, two-sided) |
|---|---|
| `self_consistency` | Within a family, instruct vs reasoning answer-divergence at T=0 differs. **Noise floor for the other two.** |
| `context_rot` | Within a family, instruct vs reasoning accuracy on a planted fact differs across the filler-turn × kind sweep. |
| `sycophancy` | Within a family, instruct vs reasoning accuracy at re-probe after wrong pushback differs across the gap × condition sweep. |

## Pivot — what's locked

This is **Pivot A** (within-family instruct ↔ reasoning comparison) — chosen
because the original "measure these pathologies cleanly" framing turned out
to be heavily covered by 2025 papers (SYCON, SycEval, "LLMs Get Lost in
Multi-Turn", non-determinism work — full table in `RELATED_WORK.md`). Pivot A
keeps the same testbed but compares across post-training regimes, which is
genuinely under-studied.

Hypotheses, primary metric, effect-size threshold, multiple-comparisons plan,
and exclusion rules are frozen in `PREREGISTRATION.md` (root) and
`experiments/<axis>/PREREGISTRATION.md` (per experiment). **Do not amend
these after looking at the data.**

## Model pairs under test

Defined in `configs/models.yaml`:

| Family | Instruct | Reasoning |
|---|---|---|
| deepseek-v | `deepseek/deepseek-v4-pro` | `deepseek/deepseek-r1-0528` |
| qwen3-235b | `qwen/qwen3-235b-a22b-2507` | `qwen/qwen3-235b-a22b-thinking-2507` |
| qwen3-30b | `qwen/qwen3-30b-a3b-instruct-2507` | `qwen/qwen3-30b-a3b-thinking-2507` |

Plus `anthropic/claude-opus-4.7` as a closed-frontier *anchor* — reported as
a separate column for context only, not part of the paired tests.

## Quickstart

```bash
cd ~/agent-pathologies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # put OPENROUTER_API_KEY=sk-or-... here

# 1) Smoke test against the mock provider — no API, no keys.
bash scripts/run_sweep.sh --mock

# 2) Dry-run cost estimate — see the bill before running.
python scripts/estimate_cost.py

# 3) Real sweep, resumable (re-running skips completed cells).
bash scripts/run_sweep.sh

# 4) Cross-experiment paired headline.
python scripts/pair_analysis.py
```

Estimated total cost for the full v2 sweep across all paired families +
Claude anchor: **~$40** (heavily dominated by the Claude anchor; ~$9 without
it). Reasoning models can spend 3–10x more if they emit long thinking
traces — budget headroom.

## What's rigorous about this

- **Pre-registered.** `PREREGISTRATION.md` locks hypotheses, primary metric,
  effect-size threshold (Cohen's h ≥ 0.20), multiple-comparisons correction
  (Benjamini-Hochberg), sample sizes, and stopping rules **before** data
  collection. Falsification criteria are explicit.
- **Paired by design.** Same `task_seed` is run on both instruct and
  reasoning members of each pair. Comparison uses **McNemar's exact** test
  on paired binary outcomes per cell, plus bootstrap CIs on proportions.
- **Strict scoring.** No LLM-judge anywhere in the eval loop. Tasks use
  regex-based answer extraction with word-boundary matching, so an
  arithmetic answer of "8" does not falsely match the operand "8" copied
  back into a model's response.
- **Control conditions.**
  - `context_rot.kind = token_matched` isolates "many turns" from "many tokens".
  - `sycophancy.condition = correct` + `= neutral` isolate generic
    capitulation from wrong-belief adoption — the interpretable signal is
    `acc(correct) − acc(wrong)`.
- **Exclusions, not silent re-sampling.** Frozen rules in
  `src/agent_pathologies/analysis/exclusions.py` flag refusals, truncation,
  unscorable answers. Counts reported separately; excluded rows are
  excluded from hypothesis testing.
- **Resumable runs.** Re-running any experiment after a crash skips cells
  already in the JSONL log (`extra.cell_key`). Mid-sweep failures are cheap.
- **Token + cost tracking.** Every trajectory carries `input_tokens`,
  `output_tokens`, `cost_usd`. `scripts/estimate_cost.py` projects total
  spend before you run.
- **38 unit tests** covering scoring, synthesizers, statistical helpers, and
  runner resumability. Run with `python -m pytest tests/`.

## Repo layout

```
configs/
  models.yaml                   # within-family pairs + anchor
  pivot_a.yaml                  # sweep parameters (n_tasks, sweeps, thresholds)

src/agent_pathologies/
  types.py                      # Trajectory, Turn, ModelRole (pydantic)
  client.py                     # Multi-provider async client (mock|openrouter|...)
  runner.py                     # Resumable trajectory executor + cost tracking
  budget.py                     # Token counting + cost estimation
  config_loader.py              # YAML → RunSpec iterator
  tasks/
    base.py                     # Task + TaskInstance with scorer callable
    scoring.py                  # Strict answer extraction
    arithmetic.py
    needle_qa.py                # Single-fact needle
    multi_fact_needle.py        # N facts planted, one probed (context_rot)
    closed_qa.py                # Multiple-choice factual QA (for follow-ups)
  conversation/
    synthesizer.py              # 4 filler kinds incl. token_matched control
    pushback.py                 # 3 pushback conditions: wrong/correct/neutral
  analysis/
    metrics.py                  # accuracy_with_ci, divergence, exclusion report
    stats.py                    # McNemar, Cohen's h, bootstrap, BH correction
    plots.py                    # paired bars, accuracy curves w/ CIs
    exclusions.py               # frozen exclusion rules

experiments/<axis>/
  run.py                        # config-driven, resumable, paired
  analyze.py                    # paired McNemar / Wilcoxon + BH + plots
  PREREGISTRATION.md            # locked hypothesis + plan
  README.md

scripts/
  run_sweep.sh                  # one-shot full sweep + analyze
  estimate_cost.py              # dry-run cost projection
  pair_analysis.py              # cross-experiment headline table

tests/                          # 38 unit tests, pytest
data/                           # JSONL trajectories + analysis CSVs + PNG plots
```

## Reproducing a finding from scratch

```bash
# A specific finding, e.g. "qwen3-235b reasoning beats instruct on context_rot at k=20":
python experiments/context_rot/run.py
python experiments/context_rot/analyze.py
# Result lives in data/context_rot.jsonl (raw) and data/context_rot_paired.csv
# (paired McNemar + Cohen's h + BH-corrected q per cell).
```

Trajectories are persisted in JSONL with `cell_key` hashes; analysis is
deterministic given the data; randomness in bootstrap CIs is seeded.

## Status

| Component | Status |
|---|---|
| Pre-registration locked | done |
| Multi-provider client | done |
| Tasks (arithmetic, needle, multi-fact-needle, closed-QA) | done |
| Control conditions (filler kinds + pushback conditions) | done |
| Resumable runner + token/cost tracking | done |
| Paired statistical analysis (McNemar, Wilcoxon, BH) | done |
| Per-experiment pre-registrations | done |
| Unit tests (38) | done |
| Mock-provider smoke test (12,500 trajectories) | done |
| Real-model sweep | awaiting OPENROUTER_API_KEY |
| Mechanistic interp follow-up (Pivot B) | future paper |
