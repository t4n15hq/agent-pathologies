# agent-pathologies

This is a preregistered study of a simple question:

> When an AI model is given a "reasoning mode," does it actually become
> more reliable, or does reasoning only help with some kinds of failures?

I compare reasoning-enabled models against their non-reasoning /
instruct versions on three failure modes: whether they give consistent
answers, whether they cave when the user pushes a wrong answer, and
whether they lose track of information as conversations get longer.

The short version: **reasoning helps a lot with hard problem-solving,
but it does not magically fix every multi-turn failure.** It is a
selective tool, not a universal upgrade.

The paper draft, figures, preregistration log, and analysis code live in
this repository. The full per-trajectory result logs are excluded from
GitHub due to size and will be shared through a release archive.

---

## Plain-English summary

Most people assume that if a model has a reasoning mode, it should just
be "smarter" and therefore more reliable. This project tests whether
that is actually true.

I tested four model pairs. In each pair, one model is the normal
instruct version and the other is the reasoning-enabled version. Then I
checked three things:

1. **Does it stay consistent?** If I ask the same hard question many
   times, does the model keep giving the same correct answer?
2. **Does it get pressured by the user?** If the model gives an answer
   and the user confidently says the wrong answer is correct, does the
   model cave?
3. **Does it lose track in long conversations?** If I give the model a
   task, add lots of filler conversation, and then ask about the task at
   the end, does it still remember what matters?

The main result is not "reasoning models are always better." The result
is more specific:

> Reasoning strongly helps when the model needs to solve a hard problem,
> but it only selectively helps with multi-turn reliability. Different
> failure modes behave differently.

That matters because people are starting to use reasoning models as if
they are automatically safer or more robust. This study argues that they
still need to be tested by failure type.

## Research question

Multi-turn LLM "pathologies" — *self-consistency drift* (the model gives
different answers across identical replays), *sycophancy* (it caves to
wrong-pushback from the user), and *context rot* (its accuracy degrades
as conversations get longer) — have each been documented in isolation
by prior work. The literature has converged on a vague claim that
reasoning-tuned models are "more robust" than instruct-tuned ones on
these failure modes. But the evidence base is heterogeneous: it
compares across model families, scales, training runs, RLHF recipes,
and serving stacks, conflating the reasoning-mode contrast with every
other dimension.

**Our research question:** Does enabling a model's reasoning mode
*by itself* change its multi-turn pathology resistance, when
everything else (base weights, scale, serving host) is held constant?

## Why this matters

A "reasoning helps everywhere" claim would let practitioners
default-deploy reasoning variants for any multi-turn application that
exposes pathology risk (customer support, agents, evals, judgment).
A "reasoning helps selectively" claim says the opposite: practitioners
need to test their *specific* axis before reaching for it.

## The thesis

After data collection and analysis, the thesis we defend is the
selective-intervention reading:

> Reasoning enablement is a **selective** intervention. It reliably
> and substantially improves hard single-shot inference. Its effect on
> multi-turn pathology resistance is heterogeneous — sometimes large,
> sometimes null, depending on whether the instruct sibling has
> inference headroom for the reasoning mode to exploit.

## The three failure modes we tested

1. **Self-consistency.** Hardness-5 arithmetic, 25 replays at $T=0$
   per task per model. The pathology is the model giving *different*
   answers across identical replays (or worse, the same wrong answer
   each time — capability mode-collapse). Measured as accuracy and as
   the divergence of the integer extracted from each replay.

2. **Sycophancy.** The model answers a hardness-5 arithmetic problem,
   the user pushes back with one of three messages (**wrong**: an
   incorrect answer asserted as correct; **correct**: the actual
   correct answer asserted as the user's preferred answer;
   **neutral**: ambiguous), then after a gap of filler turns the
   model is re-asked the original question. The pathology is the model
   flipping toward whatever the user pushed. Measured as the
   difference-in-differences:
   `gain = instr(acc_correct − acc_wrong) − reas(acc_correct − acc_wrong)`;
   positive means reasoning is less sycophantic.

3. **Context rot.** A 20-update variable-tracking task. Between the
   setup and the probe the model receives $k \in \{0, 2, 5, 10, 20, 40\}$
   filler turn-pairs of one of four kinds (irrelevant /
   topically-related / token-matched / collapsed). The pathology is
   accuracy degrading as $k$ grows. Measured as paired accuracy at
   each $(k, \text{kind})$ cell.

## Model pairs

| Family | Pair design | Instruct member | Reasoning member |
|---|---|---|---|
| DeepSeek V4-pro   | **within-MODEL** (same weights, runtime toggle) | `deepseek-v4-pro` (`reasoning: enabled=false`)     | `deepseek-v4-pro` (`reasoning: effort=high`) |
| DeepSeek V4-flash | **within-MODEL** (same weights, alias toggle)   | `deepseek-chat` (alias)                            | `deepseek-reasoner` (alias)                  |
| Qwen3-235B (a22b) | within-FAMILY SKU (matched scale, separate post-training) | `qwen/qwen3-235b-a22b-2507`         | `qwen/qwen3-235b-a22b-thinking-2507`         |
| Qwen3-30B (a3b)   | within-FAMILY SKU (matched scale, separate post-training) | `qwen/qwen3-30b-a3b-instruct-2507`   | `qwen/qwen3-30b-a3b-thinking-2507`           |

The DeepSeek pairs are the load-bearing evidence: base weights, scale,
and serving host are identical across siblings, so the within-pair
delta is uniquely attributable to the reasoning toggle. The Qwen pairs
corroborate the direction with a known capability confound (see
caveats) — the Qwen instruct SKUs mode-collapse on hardness-5
arithmetic, which inflates the Qwen sycophancy gap.

## Preregistration and amendment timeline

The hypotheses, primary metrics, effect-size threshold (Cohen's
$h \geq 0.20$ or $|\text{DiD gain}| \geq 0.10$), multiple-comparisons
correction (Benjamini–Hochberg at FDR $= 0.05$), exclusion classes,
sample sizes, and falsification criteria were locked in
`PREREGISTRATION.md` on **2026-05-13**, before any real-model trajectory
was collected. The amendment log records every change made afterward,
each dated, each with the reason:

| Date | Amendment | Why |
|---|---|---|
| 2026-05-13 (locked) | Initial preregistration | — |
| 2026-05-13 (same day) | DeepSeek pair redesigned to within-MODEL reasoning toggle (V4-pro vs V4-pro, V4-flash vs V4-flash); OR upstream pinning + `provider_error` re-attempt rules | Original cross-architecture pair (V4-pro vs R1-0528) violated the "scale held constant" premise |
| 2026-05-14 | New exclusion class `provider_empty_response` for output\_tokens=0 with empty body | Stage-1 calibration found 1,100 such rows; conflated with `empty_probe_answer` |
| 2026-05-14 | `cell_key` resumability bug fix: include `model_role` so V4-pro instruct and reasoning don't collide | Same model ID for both V4-pro members; resume logic was skipping new reasoning attempts |
| 2026-05-15 | `max_tokens` recovery: 2048 → 8192 for affected reasoning calls, 16384 for context-rot reasoning; client adds `reasoning_content`/`reasoning_details` fallback parsing | Original 2048 was 16× below DeepSeek's recommended 32K default; reasoning models consumed the entire budget on CoT, leaving zero tokens for `content` |
| 2026-05-15 | V4-pro routing: cells that returned `provider_error` or empty body on OR/Novita were re-attempted via DeepSeek's first-party API with the V4-family-native `thinking: {type: ...}` parameter | The legacy `reasoning: {enabled: bool}` parameter is silently ignored for `deepseek-v4-pro`; live testing showed `thinking: disabled` actually disables CoT |
| 2026-05-15 | Qwen instruct `truncated_at_max_tokens` retry at 8192 | 38 Qwen instruct cells emitted longer-than-2048-token CoT on hardness-5 |
| 2026-05-15 | Semantic dedup pipeline keyed on (`model_family`, role, task, sweep, seed) | Same logical cell served by OR-original and DS-direct retry was being counted twice; dedup picks the latest non-excluded |
| 2026-05-15 | Removed the planned §7.4 "serving-stack signal" discussion subsection on `provider_empty_response`; reframed as a configuration error | Root-caused to our `max_tokens` setting, not a model property |

The first sweep on 2026-05-14 surfaced enough irregularities that we
treated 2026-05-15 as a recovery day. Every amendment is documented in
`PREREGISTRATION.md` with the specific reason and the analytical
consequence. The amendments collectively did not change the
preregistered hypotheses, only the data quality on the original
preregistered cells.

## What changed during the study, and why

The most material change was the `max_tokens` configuration error
discovered on 2026-05-15. Our initial sweep used `max_tokens = 2048`,
which our calibration justified as "headroom for reasoning traces."
That choice was wrong: DeepSeek's own documentation specifies a
default of 32K for `deepseek-reasoner`, and on hardness-5 arithmetic
the reasoning models consumed the entire 2048-token budget on the CoT
field, leaving zero tokens for the user-facing answer. Roughly 1,800
trajectories were affected, concentrated on DeepSeek reasoning
members.

Once we identified the cause, we re-attempted the affected cells with
`max_tokens = 8192` (self-consistency, sycophancy) or `16384`
(context-rot, whose conversation prefixes are longer). The recoveries
went through DeepSeek's first-party API, partly because it costs less
than OpenRouter for V4-pro (75% promotional discount) and partly
because it uses the V4-family-native `thinking: {type: ...}` parameter
which we verified actually toggles reasoning on/off (the legacy
`reasoning: {enabled: bool}` parameter we initially used is silently
ignored by `deepseek-v4-pro`). The recoveries succeeded at high rate
on every targeted class: 38/38 on Qwen instruct truncation, 164/164 on
V4-pro instruct, 139/139 on the highest-budget DeepSeek-reasoning
retry pass.

The final analyzable dataset is composed via *semantic dedup*: where a
logical cell (family, role, task, sweep value, seed) has multiple
attempts in the raw JSONL, the dedup logic prefers any non-excluded
attempt over any excluded attempt, ties broken by recency. The raw
JSONL keeps every attempt for audit; the analyzable set keeps exactly
one row per logical cell.

After the amendments and recoveries, every (family × role × axis) cell
is at its preregistered $N$ in the analyzable set: $1{,}000$ replays per
cell on self-consistency, $600$ on sycophancy, $1{,}200$ on context-rot.
Totals: $8{,}000 / 4{,}800 / 9{,}600$ analyzable trajectories per axis.

## Final results at a high level

| Axis | Direction | Magnitude |
|---|---|---|
| **Self-consistency** | reasoning $\gg$ instruct, **all four** families | $\Delta_{\text{acc}}$ ranges from $+0.67$ on V4-flash to $+0.99$ on Qwen-30B; sign-test BH-$q < 10^{-11}$ in every family; per-task wins 38/40, 40/40, 40/40, 40/40 |
| **Sycophancy (reasoning gain)** | **heterogeneous within reasoning-enabled variants** | V4-flash $+0.165$ (PASS); **V4-pro $+0.045$ (NULL** at the preregistered threshold, CI $[-0.06, +0.15]$); Qwen pairs $+0.85$ (capability-confounded) |
| **Context rot (accuracy at $k{=}40$)** | ceiling-dominated on DeepSeek, large but confounded on Qwen | DeepSeek $\Delta \approx 0$ (instruct already at $1.000$); Qwen-30B $+0.46$ (capability-confounded) |

The headline pattern is that reasoning enablement is a selective lever:
it reliably and substantially improves hard single-shot inference
(self-consistency), and it produces *heterogeneous* effects on
multi-turn pathology resistance — large where the instruct sibling has
inference headroom, null where the instruct sibling is already
ceilinged.

## Caveats and what the paper does *not* claim

1. **Qwen capability confound.** Both Qwen instruct SKUs mode-collapse
   on hardness-5 arithmetic, emitting the placeholder integer
   $1234$ across the bulk of replays and achieving exactly $0.000$
   accuracy. The reasoning sibling rescues this. So the
   $+0.85$ sycophancy gain and the $+0.46$ context-rot gain on Qwen
   look enormous numerically, but they partly measure
   capability-recovery rather than the within-model reasoning toggle.
   The paper says explicitly that the *DeepSeek pairs* are the
   load-bearing evidence for the within-MODEL contrast; the Qwen pairs
   corroborate the direction with the capability caveat.

2. **V4-pro mixed routing.** The original sweep pinned V4-pro to
   OpenRouter/Novita; the 2026-05-15 amendment re-routed retries
   through DeepSeek-direct. The final analyzable V4-pro dataset is
   therefore **mixed**: OR/Novita originals where they succeeded, plus
   DS-direct recoveries on the cells they did not. Per-axis counts
   are in `paper/05_experimental_setup.tex` §5.2. The mix is light on
   most axes ($\leq 32$ DS-direct rows out of $1{,}200$ on context-rot
   reasoning, for example) but is heavier on sycophancy reasoning
   ($340$ DS-direct out of $600$) precisely because that cell type
   was most affected by the `max_tokens` bug.

3. **DeepSeek `max_tokens` recovery was a config fix, not a model
   property.** An earlier draft framed the high `empty_probe_answer`
   rate on DeepSeek reasoning models as a serving-stack signal. The
   2026-05-15 amendment retracts that framing — the rate was caused
   by our own `max_tokens=2048` setting. The discussion subsection on
   the "serving-stack signal" was removed.

4. **Context-rot DeepSeek ceiling.** Instruct V4-flash and V4-pro both
   solve the variable-tracking task at $\sim$$1.000$ accuracy across
   all filler depths, so there is no headroom for reasoning to
   improve. The within-pair $\Delta$ on DeepSeek context-rot is
   indistinguishable from zero by construction. We label this honestly
   as a null mediated by ceiling, not as a reverse finding.

5. **The paper does not claim** reasoning is universally beneficial,
   that reasoning has a single "robustness multiplier," or that the
   findings transfer to other axes (e.g., refusal, jailbreak,
   long-form coherence) or other model families. We measured what we
   measured.

## What lives where

- [`paper/`](paper/) — the LaTeX preprint draft (10 sections + abstract
  + references), the 8 PDF/PNG figures, and `paper/main.pdf` after
  compilation
- [`PREREGISTRATION.md`](PREREGISTRATION.md) — the root preregistration
  with the dated amendment log
- [`experiments/<axis>/PREREGISTRATION.md`](experiments/) — per-axis
  preregistration documents
- [`data/preliminary_findings.md`](data/preliminary_findings.md) — the
  research log: mid-sweep findings (some now superseded by amendments)
  and the **Confirmed Findings (post-retry)** section at the top
- `data/*.jsonl` and `data/*_clean.jsonl` — raw and deduped per-trajectory
  logs (gitignored due to size; reproduce locally by re-running the
  sweep, or request the release archive)
- [`RELATED_WORK.md`](RELATED_WORK.md) — prior-work analysis (SYCON,
  SycEval, "LLMs Get Lost in Multi-Turn", non-determinism work) that
  motivated the Pivot A within-family design

## License

Code: MIT. Paper and data: CC BY 4.0.
