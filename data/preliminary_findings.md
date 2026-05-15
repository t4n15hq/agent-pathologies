# Preliminary Findings — Pivot A

**Original snapshot:** 2026-05-14 06:57 CDT (mid-sweep).
**Below — confirmed findings (post-retry) added 2026-05-15.**

The sections after the Confirmed Findings block are the original
mid-sweep notes, retained for chronological audit. Numbers there are
under-sampled / pre-retry and superseded for any quantity that reappears
in a Confirmed Finding.

---

## Confirmed Findings (post-retry, 2026-05-15)

After the cell_key fix (2026-05-14), the max_tokens fix on DeepSeek
reasoning (2026-05-15), the V4-pro routing change to DeepSeek-direct
(2026-05-15), and the Qwen instruct truncated_at_max_tokens retry
(2026-05-15), the self-consistency axis is fully analyzable
($8{,}000/8{,}000 = 100\%$, $1{,}000$ replays per (family, role) cell).
Sycophancy and context-rot retries are still in flight.

### F-01 — Reasoning mode massively improves single-shot solver reliability on hardness-5 arithmetic.

Headline accuracy by family on $N=40$ tasks, $25$ replays each (paired
per task; cell sizes match exactly):

| Family    | acc instr | acc reas | $\Delta_{\mathrm{acc}}$ | 95% bootstrap CI | wins/ties/losses | sign-test BH-$q$ |
|-----------|----------:|---------:|------------------------:|:----------------:|:----------------:|------------------:|
| V4-flash  | $0.316$   | $0.990$  | $+0.674$                | $[+0.566,+0.778]$ | $38/2/0\ (n{=}40)$ | $3.6{\times}10^{-12}$ |
| V4-pro    | $0.179$   | $0.997$  | $+0.818$                | $[+0.777,+0.849]$ | $40/0/0\ (n{=}40)$ | $1.2{\times}10^{-12}$ |
| Qwen-235B | $0.000$   | $0.978$  | $+0.978$                | $[+0.952,+0.996]$ | $40/0/0\ (n{=}40)$ | $1.2{\times}10^{-12}$ |
| Qwen-30B  | $0.000$   | $0.985$  | $+0.985$                | $[+0.959,+1.000]$ | $40/0/0\ (n{=}40)$ | $1.2{\times}10^{-12}$ |

All four pairs cross the BH-corrected sign-test threshold at
$q<10^{-11}$; per-task wins are dominant or unanimous in every family.

Integer-extracted divergence across the $25$ replays:

| Family    | div instr | div reas | $\Delta_{\mathrm{div}}$ |
|-----------|----------:|---------:|------------------------:|
| V4-flash  | $0.422$   | $0.010$  | $-0.412$ |
| V4-pro    | $0.554$   | $0.003$  | $-0.551$ |
| Qwen-235B | $0.051$   | $0.022$  | $-0.029$ |
| Qwen-30B  | $0.032$   | $0.015$  | $-0.017$ |

Interpretation:

- **DeepSeek pairs** (within-MODEL reasoning toggle, same weights and
  serving host) — the cleanest controlled comparison. Reasoning lifts
  accuracy by 67–82 pts AND drives divergence to effectively zero.
  Reasoning mode is both more accurate and more stable.
- **Qwen pairs** (cross-SKU, scale-matched) — the $+97.8 / +98.5$ pt
  accuracy gaps look larger than DeepSeek's, but the instruct baseline
  is $0.000$ because Qwen instruct mode-collapses to a single
  incorrect stock answer (most often ``The answer is 1234.''; see
  Fig 6). The divergence test alone would have missed this (low div
  with $0\%$ accuracy = consistently wrong, not noise). Treat Qwen as
  *capability-recovery* evidence; do not stack it numerically with the
  DeepSeek comparison.
- **Methodological note** — the preregistered accuracy-paired test
  caught the Qwen mode-collapse that the divergence-only test would
  have missed. This was the explicit purpose of adding accuracy as a
  co-primary measure (see PREREGISTRATION §4 and earlier §1c in this
  document).

Written into `paper/06_results.tex` §6.1 with both tables and a
mode-collapse paragraph; reviewer-defendable as the strongest single
result in the paper.

### F-02 — DeepSeek-reasoning empty-response rate was a configuration error, not a serving-stack signal.

Earlier hypothesis in §3 below (1,091 `empty_probe_answer` rows on
DeepSeek reasoning models, with `output_tokens=0`) was diagnosed as a
provider-side signal and explicitly **disconfirmed** on 2026-05-15.

Root cause: our `max_tokens=2048` setting was below the
DeepSeek-recommended default of 32K for `deepseek-reasoner`. On
hardness-5 arithmetic the model consumed the entire 2048-token budget
on `reasoning_content` (CoT), leaving zero tokens for `content` (final
answer). Live test reproduced the empty content 5/5 times at 2048; at
$\mathrm{max\_tokens}{=}8192$ the same prompt returned the correct
answer (16141) cleanly. PREREGISTRATION amendment 2026-05-15 documents
the fix and the V4-pro routing change to DeepSeek-direct (which moved
both V4-pro pair members onto the same first-party API and exploited
the 75\% promotional discount).

The original §7.4 "serving-stack signal" outline in the discussion
is removed.

---

## Original mid-sweep findings (2026-05-14)

**Original snapshot:** 2026-05-14 06:57 CDT.
**Row counts at snapshot:** `self_consistency.jsonl` = 8,101; `context_rot.jsonl` = 5,407; `sycophancy.jsonl` not yet produced.
The sweep is continuing to write to `data/*.jsonl`; this report analyzes the snapshot only.

---

## TL;DR

1. **`deepseek-v4-flash / instruct` divergence is partly metric artifact.** Raw-string divergence is 0.88; divergence on the *extracted last integer* (what the scorer uses) is 0.42. CoT chains get truncated at slightly different char lengths across replays; underlying integers cluster tightly (median |top1 − correct| = 1; top-1 == correct in 17/40 tasks). v4-pro and Qwen emit short canonical replies, so raw == int divergence for them.
2. **Wilcoxon-on-divergence misses the Qwen story.** Both Qwen instruct SKUs achieve **0.00 accuracy** with low divergence — consistent wrong answers (often literal `"The answer is 1234."`). Paired Wilcoxon on per-task accuracy: q = 4.7e-9, Δ = −98 pp.
3. **`empty_probe_answer` is not max_tokens truncation.** All 1,091 rows have `output_tokens = 0` and empty assistant content. Provider returned no body. Concentrated in `deepseek-reasoner` (668) and `deepseek/deepseek-v4-pro reasoning` (407). Bumping max_tokens will not help.
4. **Context-rot (v4-flash only) is directionally weak but consistent.** 15/24 cells favor instruct, 0/24 favor reasoning, 9/24 tied; |h| ≥ 0.20 in 15/24 cells but min BH-q = 1.00 — n = 6–18 per cell, underpowered. Pool cells.
5. **Qwen instruct fingerprint:** the literal `"The answer is 1234."` appears in 68% of `qwen3-30b` and 33% of `qwen3-235b` analyzable rows. Zero correct answers across 1,664 analyzable Qwen instruct rows.

---

## 1. Self-consistency — paired tests

### 1a. Verification of divergence numbers (Q1)

Re-computed paired Wilcoxon on per-task divergence. Reproduces the user's CSV exactly:

| Family            | n_paired | div_instruct | div_reasoning | Δ      | wilcoxon_p | q_BH    |
|-------------------|---------:|-------------:|--------------:|-------:|-----------:|--------:|
| deepseek-v4-flash | 35       | 0.882        | 0.019         | +0.863 | 2.0e-7     | 8.1e-7  |
| deepseek-v4-pro   | 31       | 0.533        | 0.000         | +0.533 | 1.7e-6     | 3.5e-6  |
| qwen3-235b        | 40       | 0.049        | 0.023         | +0.026 | 3.1e-2     | 4.1e-2  |
| qwen3-30b         | 40       | 0.031        | 0.015         | +0.016 | 1.5e-1     | 1.5e-1  |

### 1b. Divergence is partly metric artifact (Q3)

`answer_divergence()` operates on raw `probe_answer` strings. Models that emit short canonical replies ("The answer is N.") get raw == integer-level divergence; models that emit CoT before a final integer get string-level divergence inflated by char-length variation in the CoT. Divergence on the **last-integer extraction** (what the scorer uses):

| Family            | role      | div (raw string) | div (last int) |
|-------------------|-----------|-----------------:|---------------:|
| deepseek-v4-flash | instruct  | 0.880            | **0.422**      |
| deepseek-v4-flash | reasoning | 0.019            | 0.019          |
| deepseek-v4-pro   | instruct  | 0.495            | 0.495          |
| deepseek-v4-pro   | reasoning | 0.000            | 0.000          |
| qwen3-235b        | instruct  | 0.049            | 0.049          |
| qwen3-235b        | reasoning | 0.023            | 0.021          |
| qwen3-30b         | instruct  | 0.031            | 0.031          |
| qwen3-30b         | reasoning | 0.015            | 0.015          |

Only `deepseek-v4-flash / instruct` is affected. Spot-check of `arith-h5-31` (correct = 13293) shows the underlying integer distribution clusters around the right answer:

```
top-1 integers extracted over 25 replays:
  13293 × 11   ← correct
  13292 ×  8
  12492, 12892, 12493, 12553, 12653, 12893 × 1 each
```

raw-string divergence = 0.96 (every CoT got truncated at a slightly different word) but integer divergence = 0.56 with the modal answer being *correct*.

Aggregated for `deepseek-v4-flash` instruct over all 40 tasks:
- mean unique integers per task: 5.2 (median 5)
- mean top-1 share: 0.578 (top-5 share: 0.951 — answers cluster)
- tasks where top-1 == correct: **17/40**
- median |top1 − correct|: **1**

Interpretation: this arm isn't randomly guessing — it's making small, tightly clustered arithmetic errors and getting cut off mid-reasoning. The paper should report divergence on the last-integer extraction (or on accuracy) rather than on the raw string. Repeating the paired Wilcoxon on int-divergence:

| Family            | n  | div_int_instr | div_int_reas | Δ      | p        | q_BH     |
|-------------------|----|--------------:|-------------:|-------:|---------:|---------:|
| deepseek-v4-flash | 35 | 0.409         | 0.019        | +0.390 | 9.5e-7   | 3.2e-6   |
| deepseek-v4-pro   | 31 | 0.533         | 0.000        | +0.533 | 1.7e-6   | 3.2e-6   |
| qwen3-235b        | 40 | 0.049         | 0.021        | +0.027 | 2.9e-2   | 3.9e-2   |
| qwen3-30b         | 40 | 0.031         | 0.015        | +0.016 | 1.5e-1   | 1.5e-1   |

The qualitative conclusion (reasoning more consistent) still holds at q < 0.05 for three families; effect on v4-flash is roughly halved.

### 1c. Complementary paired test on accuracy (Q2)

Per-task `is_correct` mean (paired Wilcoxon across tasks):

| Family            | n  | acc_instruct | acc_reasoning | Δ (pp) | wilcoxon_p | q_BH    |
|-------------------|----|-------------:|--------------:|-------:|-----------:|--------:|
| deepseek-v4-flash | 35 | 0.328        | 0.981         | −65.3  | 4.6e-7     | 4.6e-7  |
| deepseek-v4-pro   | 31 | 0.005        | 1.000         | −99.5  | 4.1e-8     | 5.4e-8  |
| qwen3-235b        | 40 | 0.000        | 0.979         | −97.9  | 2.4e-9     | 4.8e-9  |
| qwen3-30b         | 40 | 0.000        | 0.985         | −98.5  | 9.1e-10    | 3.6e-9  |

**Both Qwen families are at q < 1e-8 on the accuracy test even though divergence Δ is small (or fails the divergence test entirely for qwen3-30b).** The two tests are complementary:

- **Divergence test catches:** "the model gives different answers across replays" → flags v4-flash / v4-pro / qwen3-235b.
- **Accuracy test catches:** "the model gives the same wrong answer across replays" → flags both Qwens (and confirms the DeepSeek pair).

For the paper, recommend pre-registering the accuracy paired Wilcoxon as a co-primary test alongside divergence, so consistently-wrong mode-collapse isn't a false-negative.

### 1d. Spot-checks

- `deepseek-v4-flash / arith-h5-6` (correct = −7397): instruct (n=25) emits a chain that never reaches an integer; reasoning (n=3) emits `'the answer is -7397.'` × 3.
- `qwen3-30b / arith-h5-0` (correct = 9824): instruct (n=18) emits `'the answer is 1234.'` × 18; reasoning (n=19) emits `'the answer is 9824.'` × 19.

---

## 2. Context-rot — within-pair gap, deepseek-v4-flash only (Q4)

Only the `deepseek-v4-flash` pair has analyzable context-rot data in the snapshot; all other pairs are 100% `provider_error:RetryError` (likely re-collecting after the OpenRouter credit-exhaustion noted in the 2026-05-14 amendment).

Marginal pattern: instruct accuracy is **1.000 in every cell**; reasoning accuracy is 0.83–1.00. Paired McNemar across 24 (kind × n_filler) cells:

| Statistic                                       | Value         |
|-------------------------------------------------|---------------|
| Cells with delta < 0 (instruct > reasoning)     | 15 / 24       |
| Cells with delta > 0 (reasoning > instruct)     | 0 / 24        |
| Cells tied                                      | 9 / 24        |
| max |h|                                         | 0.841         |
| Cells with |h| ≥ 0.20 (preregistered threshold) | 15 / 24       |
| min BH-q                                        | 1.000         |
| Range of Δ (reasoning − instruct)               | [−0.167, 0]   |

Every directional cell points the same way (instruct slightly more robust), but per-cell n is 6–18 so McNemar's b+c is too small to clear q < 0.10 on any one cell. This is consistent with either (a) a real but small effect, or (b) noise with one-sided distribution due to a ceiling at 1.0.

The "kinds" don't separate cleanly: reasoning accuracy at n_filler = 40 is 0.96 (collapsed), 0.89 (irrelevant), 0.91 (related), 0.94 (token_matched) — no monotone "related filler is worst" pattern.

**Recommendation:** when the OpenRouter cells finish, pool the 24 cells per family into a single McNemar (or fit a logistic mixed model on `is_correct ~ role + n_filler + kind + (1|task_id)`). Per-cell McNemar in isolation will likely never clear FDR with n ≤ 20.

---

## 3. Exclusion forensics — `empty_probe_answer` (Q5)

User hypothesis: reasoning traces truncated at max_tokens=2048. **Hypothesis is wrong.**

| model                          | upstream         | n   |
|--------------------------------|------------------|-----|
| `deepseek-reasoner`            | `deepseek_direct`| 668 |
| `deepseek/deepseek-v4-pro` (reasoning) | `Novita`  | 407 |
| `deepseek/deepseek-v4-pro` (instruct)  | `Novita`  | 16  |

**For all 1,091 rows:** `output_tokens = 0`, `input_tokens ≈ 111`, last assistant turn has empty content. No content was generated at all — the provider returned an empty response (and was still billed input tokens). Sample row:

```
output_tokens: 0, input_tokens: 111, cost_usd: 4.77e-05, error: None
upstream_actual: Novita,  upstream_observed_all: ['Novita']
turns: [system, user, assistant(content='')]
```

This is provider-side: either silent content filtering, a token-stream that closed before any visible content, or a server-side error that surfaced as `200 OK` with empty content. Worth confirming with a packet-capture or by querying the provider, but it's clearly **not** a max_tokens issue — bumping max_tokens won't help.

Implication: `analysis/exclusions.py` conflates "provider returned empty body" with "model produced unscorable text". Recommend a new reason `provider_empty_response` (output_tokens == 0 AND assistant content == ""). The 2026-05-14 amendment about resumability only re-runs `provider_error:*`, so these cells are NOT being re-attempted — they're locked as analyzable-excluded.

Seed-level failure rate for `deepseek-v4-flash` reasoner is 0.08–0.96 (mean 0.67), clustering by prompt rather than perfectly random. Of 40 tasks, 36 have ≥ 1 successful reasoner reply, with replays ranging 1–23 (median 7.5). The reasoner pair is severely under-sampled.

Also exactly **1 `unscorable_answer`** row: `deepseek-v4-pro / reasoning / arith-h5-0`, `probe_answer = 'The answer'` (content length 10, output_tokens = 2) — a different mid-sentence truncation.

---

## 4. Cost / sample-size diagnostic (Q6)

### `self_consistency.jsonl` (total cost $0.294)

| Family            | role      | n_total | n_analyz | excl_rate | cost ($) | mean in_tok | mean out_tok | exclusion breakdown |
|-------------------|-----------|--------:|---------:|----------:|---------:|------------:|-------------:|---------------------|
| deepseek-v4-flash | instruct  | 1000    | 1000     | 0.000     | 0.115    | 111         | 354          | —                   |
| deepseek-v4-flash | reasoning | 1000    | 332      | 0.668     | 0.016    | 111         | 2            | empty_probe_answer=668 |
| deepseek-v4-pro   | instruct  | 1007    | 822      | 0.184     | 0.053    | 111         | 6            | provider_error=169; empty_probe_answer=16 |
| deepseek-v4-pro   | reasoning | 1007    | 393      | 0.610     | 0.050    | 111         | 3            | empty_probe_answer=407; provider_error=206; unscorable=1 |
| qwen3-235b        | instruct  | 1029    | 846      | 0.178     | 0.009    | 111         | 6            | provider_error=183  |
| qwen3-235b        | reasoning | 1012    | 816      | 0.194     | 0.028    | 111         | 7            | provider_error=196  |
| qwen3-30b         | instruct  | 1023    | 818      | 0.200     | 0.012    | 111         | 6            | provider_error=205  |
| qwen3-30b         | reasoning | 1023    | 838      | 0.181     | 0.011    | 111         | 6            | provider_error=185  |

Note: `mean out_tok` of 2–9 for reasoning roles is misleading — DeepSeek's API and OpenRouter's Qwen-thinking SKUs do not surface internal reasoning tokens; the `output_tokens` column reflects only the visible final answer. The actual thinking budget is hidden.

### `context_rot.jsonl` (total cost $4.141)

| Family            | role      | n_total | n_analyz | excl_rate | cost ($) | mean in_tok | mean out_tok |
|-------------------|-----------|--------:|---------:|----------:|---------:|------------:|-------------:|
| deepseek-v4-flash | instruct  | 644     | 644      | 0.000     | 2.086    | 21,536      | 801          |
| deepseek-v4-flash | reasoning | 660     | 593      | 0.102     | 1.994    | 20,234      | 674          |
| deepseek-v4-pro   | instruct  | 679     | 0        | 1.000     | 0.021    | 73          | 0            |
| deepseek-v4-pro   | reasoning | 680     | 0        | 1.000     | 0.021    | 73          | 0            |
| qwen3-235b        | instruct  | 681     | 0        | 1.000     | 0.003    | 73          | 0            |
| qwen3-235b        | reasoning | 694     | 0        | 1.000     | 0.008    | 73          | 0            |
| qwen3-30b         | instruct  | 674     | 0        | 1.000     | 0.004    | 73          | 0            |
| qwen3-30b         | reasoning | 695     | 0        | 1.000     | 0.004    | 73          | 0            |

All OpenRouter rows for context_rot have `input_tokens ≈ 73` and zero analyzable trajectories. These are the credit-exhaustion `provider_error:RetryError` rows that will be re-attempted (per the 2026-05-14 amendment). Resume on those is what the sweep is doing right now.

### Replay coverage per task (self_consistency)

Target N is 25 replays per task per (family, role). Actual:

| Family            | role      | n_tasks | mean replays | min | max |
|-------------------|-----------|--------:|-------------:|----:|----:|
| deepseek-v4-flash | instruct  | 40      | 25.0         | 25  | 25  |
| deepseek-v4-flash | reasoning | 40      | **8.3**      | 1   | 23  |
| deepseek-v4-pro   | instruct  | 40      | 20.5         | 6   | 24  |
| deepseek-v4-pro   | reasoning | 36      | 10.9         | 1   | 23  |
| qwen3-235b        | instruct  | 40      | 21.2         | 16  | 25  |
| qwen3-235b        | reasoning | 40      | 20.4         | 16  | 24  |
| qwen3-30b         | instruct  | 40      | 20.5         | 16  | 24  |
| qwen3-30b         | reasoning | 40      | 21.0         | 17  | 24  |

The DeepSeek reasoning arms are heavily under-sampled. Divergence estimates for tasks with only 1–3 surviving replays are noisy: per the analyzer, tasks with < 3 replays are filtered, but 4–5 replays still give a coarse divergence estimate.

---

## 5. Anomaly hunt (Q7)

### 5a. The `"The answer is 1234."` mode collapse on Qwen instruct

Across all 1,664 analyzable Qwen instruct rows in self_consistency:

| Family + upstream         | n   | pct emitting "...1234." |
|---------------------------|----:|------------------------:|
| qwen3-30b / Nebius        | 818 | **67.6%**               |
| qwen3-235b / Novita       | 846 | **32.7%**               |

Top probe_answer strings for `qwen3-30b/instruct`:
```
553 × "the answer is 1234."   ← literal placeholder
 23 × "the answer is 2764."
 22 × "the answer is 4867."
 22 × "the answer is 8509."
 20 × "the answer is -8984."
```

Overall accuracy of Qwen instruct on hardness-5 arithmetic = **0.000** (0 / 1,664). This isn't a divergence problem — it's a competence + mode-collapse problem at this hardness. The same prompts to the *thinking* SKU yield 0.98 accuracy. Two reads:

1. The Qwen instruct SKUs effectively cannot solve hardness-5 problems and back off to a memorized placeholder integer ("1234"). The placeholder bias differs by upstream (Nebius for the 30B vs Novita for the 235B) and by SKU. The cross-upstream-host serving stack confound is documented in the preregistration but the *direction* of failure is consistent: both Qwen instruct arms catastrophically fail at h=5.
2. Combined with the v4-flash instruct showing 0.33 accuracy where v4-flash reasoning is 0.98, **reasoning training appears to be doing most of the work at this hardness** — but the proper paper-level claim should be about the **accuracy delta**, not the divergence delta, because divergence underrates this. (See §1c.)

### 5b. Upstream provider distribution

Self-consistency upstream_actual seen:
- `Novita` 2,485 (Qwen 235B + v4-pro)
- `deepseek_direct` 2,000 (the v4-flash pair)
- `NaN` 1,144 (mostly provider_error rows that never got a response)
- `AtlasCloud` 838 (probably one of the qwen3-30b roles)
- `Nebius` 818 (qwen3-30b instruct)
- `Alibaba` 816 (qwen3-30b reasoning, judging by row counts)

Zero `upstream_mismatch` exclusions in either experiment — the pinning is working as advertised. Good.

### 5c. Long-output without scorable answer

Only **1** unscorable_answer row in self_consistency: `deepseek/deepseek-v4-pro reasoning arith-h5-0`, `probe_answer = "The answer"` (truncated mid-sentence at output_tokens = 2). Rare.

### 5d. Truncation of v4-flash instruct CoT

All 1,000 `deepseek-v4-flash / instruct` analyzable rows end with a sentence terminator (`.`, `!`, `?`), so the truncation rule (`output_tokens >= max_tokens AND no terminator`) doesn't fire. But the CoT content shows the model never reaches the final integer — it stops mid-step with a period that closes an intermediate computation. `extract_last_integer` therefore pulls an intermediate result. `output_tokens` for these rows: min 168, mean 354, max 740 — so the cap is somewhere ≥ 740 (not 512).

### 5e. v4-pro pair: instruct ≈ 0% correct

`deepseek/deepseek-v4-pro` with reasoning disabled gets **0.001** accuracy (1 / 822 correct, on a single edge-case task). The same model with reasoning enabled gets 1.000. This is the cleanest within-model evidence that the reasoning toggle is doing real work at hardness 5; no architecture/scale/upstream confound at all.

---

## 6. Recommended next steps

1. **Add a `provider_empty_response` exclusion reason** in `exclusions.py`: `if output_tokens == 0 AND assistant content == "": return "provider_empty_response"`. Distinguishes provider-side failures from model behavior. Bumping max_tokens **will not fix** the 1,091 empty-body events; do not change that knob for this reason.
2. **Pre-register the accuracy paired Wilcoxon as a co-primary self_consistency test alongside divergence.** The divergence test alone calls qwen3-30b a null result while the model is consistently emitting a wrong placeholder. This is exactly the false-negative the preregistration §8 falsification rule needs to guard against.
3. **Report self_consistency divergence on the extracted last-integer**, not the raw `probe_answer` string. Change `answer_divergence()` to accept a normalizer, or compute it post-extract. The v4-flash instruct divergence drops from 0.88 to 0.42, which is a more honest "answer instability" number. Document this in the analyzer and as a non-frozen reporting choice.
4. **Tighten the truncation rule.** Many `deepseek-v4-flash / instruct` rows end in a period but never finish the computation. Consider augmenting the rule: if the response contains intermediate-step markers like "then", "next", "step", and lacks a final "the answer is" / "= N" pattern, treat as truncated. Or simply require the response to contain at least one "answer is" / "= N" / final-line pattern.
5. **For context_rot, pool cells before testing**, given per-cell n = 6–18. A logistic mixed model with random intercepts per `task_id` and fixed effects for `role × n_filler × kind` would be more powerful than 24 separate McNemars. The current 15/24 unidirectional pattern is suggestive but not significant at the cell level.
6. **Investigate the Qwen "1234" fingerprint.** Worth a quick targeted probe: re-run a handful of hardness-3 prompts to confirm the instruct SKUs do solve easier problems (so this is a hardness-induced fallback, not a broken endpoint). If the placeholder appears at h=3 too, the Qwen instruct arm may be effectively unusable for this benchmark.
7. **Don't backfill the under-sampled DeepSeek reasoner replays.** The frozen rule in §6 forbids it. Instead, lower the `≥ 3 replays` threshold in the analyzer to `≥ 5` to drop the noisiest divergence estimates, or report per-task replay counts alongside divergence.
8. **Cost budget for the rest of the sweep is comfortable.** Self-consistency totaled $0.29 and context_rot $4.14 in the snapshot. The blocked OpenRouter context_rot cells (≈ 2,700 trajectories at ~$1–2 each based on the v4-flash run) will cost ~$4–8 more once they re-attempt; well within reason.

---

## Appendix — sanity checks

Distinct model strings per (family, role): `deepseek-v4-flash` instruct = `deepseek-chat`, reasoning = `deepseek-reasoner`; `deepseek-v4-pro` instruct and reasoning both = `deepseek/deepseek-v4-pro` (single model string, request-time toggle only — cleanest within-model design); Qwen pairs are the documented instruct vs thinking SKUs. Zero `upstream_mismatch` exclusions in either experiment.
