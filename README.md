# agent-pathologies

A controlled, multi-model framework for measuring **trajectory pathologies** in
multi-turn LLM agents. Three experiments share one testbed:

| Experiment | Hypothesis |
|---|---|
| `self_consistency` | At T=0 the same agent on the same task should give the same answer. Quantify when it doesn't — this is the **noise floor**. |
| `context_rot` | Accuracy on a planted fact degrades as filler turns pile up between plant and probe, even when the fact is still in-context. |
| `sycophancy` | After a wrong user pushback the agent flips. How long does the wrong belief persist across subsequent neutral turns? |

## Models under study

Primary spine: **Chinese open-weight families** — Qwen-3, DeepSeek-V3, GLM-4.5.
Comparison anchor: **one closed frontier model** (Claude or GPT) so findings
have a published reference point.

Recommended access path: **OpenRouter** — one key, one wire format, all five
models. (Bypass it with `--provider together|fireworks|anthropic|...` if you'd
rather use a single provider directly.)

For the mechanistic follow-up on `context_rot` you'll need local open weights
(vLLM / `transformers`) — APIs don't expose attention. That's a stage-2 add-on,
not blocking the headline experiments.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # drop in OPENROUTER_API_KEY

# Smoke test against the mock provider (no keys, no network).
python experiments/self_consistency/run.py
python experiments/self_consistency/analyze.py

# Real run via OpenRouter (model IDs at openrouter.ai/models).
python experiments/self_consistency/run.py \
    --provider openrouter \
    --model "qwen/qwen-2.5-72b-instruct" \
    --n-tasks 30 --n-repeats 20
```

### Recommended model sweep (verified against OpenRouter, May 2026)

| Slot | Model ID | Context | $/M in | $/M out |
|---|---|---:|---:|---:|
| Qwen flagship | `qwen/qwen3.5-397b-a17b` | 262k | 0.39 | 2.34 |
| Qwen budget | `qwen/qwen3.5-122b-a10b` | 262k | 0.26 | 2.08 |
| Qwen reasoning | `qwen/qwen3-235b-a22b-thinking-2507` | 131k | 0.15 | 1.50 |
| DeepSeek flagship | `deepseek/deepseek-v4-pro` | 1M  | 0.43 | 0.87 |
| DeepSeek reasoning | `deepseek/deepseek-r1-0528` | 164k | 0.50 | 2.15 |
| GLM | `z-ai/glm-4.7` (budget) or `z-ai/glm-5.1` (flagship) | 203k | 0.40 / 1.05 | 1.75 / 3.50 |
| Kimi | `moonshotai/kimi-k2.6` | 262k | 0.74 | 3.50 |
| Frontier anchor | `anthropic/claude-opus-4.7` | 1M | 5.00 | 25.00 |
| Frontier reasoning anchor | `openai/gpt-5.5` | 1M | 5.00 | 30.00 |

Always re-verify at `openrouter.ai/models` — IDs and prices update.

## Repo layout

```
src/agent_pathologies/
  types.py            Trajectory, Turn (pydantic)
  client.py           Multi-provider async LLM client (+ Mock)
  runner.py           Trajectory executor and JSONL logger
  tasks/              Verifiable tasks (ground truth, no LLM judges)
  conversation/       Filler synthesizers, pushback injection
  analysis/           Metrics and plots
experiments/
  self_consistency/   Noise floor — run this first
  context_rot/        Headline finding
  sycophancy/         Companion finding
```

## Design decisions

- **Verifiable tasks only.** Probe answers are checked with substring match
  against a known ground truth. No LLM judges anywhere in the eval loop.
- **JSONL logs are the source of truth.** Re-running is for sanity checks,
  not data recovery. Trajectories land in `data/`.
- **Shared infra, separate experiments.** Each experiment has its own folder,
  README, and runnable scripts. The decision to bundle into one paper vs.
  split into three is deferred until `self_consistency` gives a noise floor
  and `context_rot` gives a primary effect size.
- **Mock provider by default.** All experiments run end-to-end without API
  keys, so infra bugs surface before you spend a dollar.

## Status

| Component | Status |
|---|---|
| Shared core (types, client, runner) | done |
| Tasks (arithmetic, needle_qa) | done |
| Conversation perturbations | done |
| `self_consistency` | runnable |
| `context_rot` | runnable |
| `sycophancy` | runnable |
| Mechanistic attention probing | TODO (open-weights only) |
| Cross-model sweep harness | TODO |
