# Pre-Registration — Pivot A

**Locked:** 2026-05-13. **Author:** Tanishq (t4n15hq). **Frozen sections:** §1–§6
below. Any deviation must be documented as an amendment with a date and reason.

The purpose of this document is to commit to the hypothesis, model set,
analysis plan, and stopping rules *before* any data is collected, so the
paper cannot be retroactively shaped to fit the data (HARKing).

---

## §1 Research question

> Do reasoning-tuned LLMs exhibit different multi-turn trajectory pathologies
> than their instruct-tuned siblings, when scale and base architecture are
> held constant?

Three pathology axes are measured: self-consistency drift, context rot,
sycophancy persistence. Each is operationalized in `experiments/<axis>/`.

## §2 Pairs under test

Two pair designs are used (see `configs/models.yaml`):

**Within-MODEL reasoning toggle (DeepSeek V4 family):**

1. `deepseek-v4-pro` instruct (reasoning disabled) ↔ `deepseek-v4-pro` reasoning (reasoning enabled, high effort)
2. `deepseek-v4-flash` instruct (reasoning disabled) ↔ `deepseek-v4-flash` reasoning (reasoning enabled)

Both members of each pair are the **same model weights** served by DeepSeek
direct (`api.deepseek.com`). The only runtime difference is the `reasoning`
parameter sent in the request. This is the strongest controlled comparison
possible — base architecture, scale, training run, and serving host are all
held constant; only the reasoning toggle varies.

**Within-FAMILY SKU pair (Qwen3):**

3. `qwen3-235b-a22b-2507` (instruct) ↔ `qwen3-235b-a22b-thinking-2507`
4. `qwen3-30b-a3b-instruct-2507` ↔ `qwen3-30b-a3b-thinking-2507`

Alibaba ships separate instruct and thinking SKUs for the same base family
and parameter count. These are served via OpenRouter with `upstream_provider`
pinned per model (Novita / Alibaba / Nebius / AtlasCloud — see YAML).
Note: within each Qwen pair, instruct and thinking variants are routed to
different default upstream hosts, which is documented as a soft limitation
of the pairing's serving-stack equivalence.

All OpenRouter requests are **pinned to a single upstream provider** per
model (`configs/models.yaml: upstream_provider`) with `allow_fallbacks: false`.
Without pinning, OpenRouter load-balances across upstream hosts, which would
silently confound the paired test.

Closed-frontier anchor (no pair): `anthropic/claude-opus-4.7` (instruct-style).
The anchor is **not** part of the paired test; it's reported as a single column
for context only.

## §3 Hypotheses

For each axis × each pair we test a directional pair:

| Axis | H₀ | H₁ (the finding we'd publish either way) |
|---|---|---|
| self_consistency | No within-pair difference in answer divergence | reasoning > instruct divergence (or <) |
| context_rot | No within-pair difference in accuracy decay slope vs. n_filler | reasoning more (or less) robust to filler |
| sycophancy | No within-pair difference in accuracy at re-probe after wrong pushback | reasoning more (or less) resistant |

The directionality is intentionally two-sided: we expect either direction is
interesting and we don't bias the protocol toward one.

## §4 Primary metric and effect-size threshold

- **Primary metric:** `is_correct` at the designated probe turn (binary) for
  `context_rot` and `sycophancy`; answer-divergence across identical replays
  for `self_consistency`.
- **Statistical test:** McNemar's exact test for paired binary outcomes in
  `context_rot` and `sycophancy`, computed per (pair × sweep-cell) and pooled
  across task instances. `self_consistency` uses paired Wilcoxon over
  per-task divergence.
- **Effect-size threshold:** **Cohen's h ≥ 0.20** on paired proportions. Below
  this we call the result a null finding *even if p < 0.05* (publication of
  null is fine; we'll report it).
- **Multiple-comparisons correction:** Benjamini-Hochberg across all pair ×
  cell tests within an axis, with FDR target = 0.05.

## §5 Sample size and stopping rule

Sample sizes are set in `configs/pivot_a.yaml` (n_tasks per axis). They are
chosen to give at least 80% power to detect h = 0.25 at α = 0.05, assuming a
correlation of 0.4 between paired outcomes. **No optional stopping**: we run
the full N and only then analyze.

## §6 Exclusions and quality controls

A trajectory is excluded from analysis if **and only if**:

1. The provider returned an HTTP error after retries (logged in `extra.error`).
2. The model refused to answer (e.g., safety refusal) — detected by absence of
   any digit in the probe answer for arithmetic tasks, or absence of any
   plausible value for needle tasks. Detection rule is in
   `src/agent_pathologies/analysis/exclusions.py` and frozen with this doc.
3. Output exceeded `max_tokens` (truncated). Detected by missing terminating
   punctuation **and** length == max_tokens.
4. **Upstream-provider mismatch.** The OpenRouter response's reported
   `provider` (header `x-openrouter-provider` or body `provider`) does not
   match the pinned `upstream_provider` from `configs/models.yaml`. This
   means a fallback fired despite `allow_fallbacks: false` and the paired
   test's "same upstream host across both members of the pair" premise is
   violated. Frozen alongside the other three rules.

Excluded trajectories are reported as a separate count per cell. We will
**not** silently re-sample to backfill exclusions.

## §7 Reporting (non-frozen)

The final paper will report, per axis:
- A paired bar chart per family (instruct vs reasoning) with bootstrap 95% CIs.
- A table of McNemar p-values and Cohen's h per (pair × cell), with BH-adjusted
  q-values.
- The anchor model's bare numbers in a separate row, not in the paired test.
- All exclusions with reasons.
- All raw JSONL trajectories will be released alongside the paper.

## §8 What would falsify the headline?

If McNemar across the three pairs returns q > 0.05 for **all** three axes and
|h| < 0.20 for **all** pairs × cells, we conclude that reasoning training does
**not** systematically alter multi-turn pathology resistance. This is itself a
publishable null result and **the paper does not get reframed** post-hoc into
"a survey of pathologies on Chinese open-weight models" — that would be an
amendment, dated, and require a fresh sweep.

---

## Amendment log

- **2026-05-13:** Updated task operationalizations before real-model data:
  self-consistency now uses arithmetic hardness 3 with fixed replay seed;
  context-rot now uses variable tracking; sycophancy now uses CRT-style
  counterintuitive math with intuitive-wrong pushback. Existing data at this
  point is mock-provider smoke data only.
- **2026-05-13 (later):** Validity hardening before real-model data:
  (a) DeepSeek pair marked exploratory because the two members differ in
  base architecture and scale, so it is excluded from the primary paired
  claim and reported as secondary;
  (b) OpenRouter upstream-provider pinning is now required per model
  (`configs/models.yaml: upstream_provider`) and exclusion rule §6.4
  is added to drop any trajectory whose actually-served upstream differs
  from the pinned one.
- **2026-05-13 (further):** DeepSeek pair redesigned. The original
  `v4-pro` ↔ `r1-0528` pairing (marked exploratory above) is dropped in
  favor of TWO within-MODEL reasoning-toggle pairs: `v4-pro` ↔ `v4-pro`
  and `v4-flash` ↔ `v4-flash`, with the `reasoning` parameter as the only
  variable. Both halves of each pair share identical weights, scale, and
  serving host (DeepSeek direct). The exploratory caveat is dropped — the
  new design is a strictly controlled within-model comparison. The DeepSeek
  pricing during this period benefits from a 75% promotional discount on
  V4-pro through 2026-05-31 and provider-side prompt caching that makes
  repeated identical prompts ~free.
- **2026-05-14:** Clarification of exclusion-rule §6 semantics under
  resumability. Genuine model-behavior exclusions (refusal_detected,
  truncated_at_max_tokens, unscorable_answer, upstream_mismatch) continue
  to count as attempted and are reported as-is — never silently
  re-sampled. **Infrastructure-only exclusions (`provider_error:*`) are
  different**: the request never reached the model. On resume these cells
  are re-attempted, so that a transient HTTP failure or a credit-exhaustion
  402 cannot permanently corrupt a cell. This treatment was applied after
  an OpenRouter credit exhaustion during stage 1 produced ~4,100 such
  exclusions in `context_rot`; they were re-collected on the subsequent
  resume run.
- **2026-05-14 (later):** Added a sixth exclusion class
  `provider_empty_response` to §6, distinguished from `empty_probe_answer`
  by `output_tokens == 0`. Stage-1 calibration data showed ~1,100 of the
  originally-labeled `empty_probe_answer` rows were actually cases where
  the provider returned HTTP 200 with an empty body and zero completion
  tokens — concentrated on `deepseek-reasoner` (668 rows) and
  `deepseek-v4-pro reasoning` via Novita (407 rows). These are NOT
  retried on resume (they appear deterministic; retrying would burn
  budget without changing outcomes), but they are counted as a distinct
  exclusion class in the paper so reviewers can see this is a serving-
  stack phenomenon, not refusals/truncations or scoring failures.
  Legacy rows tagged `empty_probe_answer` are retroactively reclassified
  by `analysis/metrics.py::reclassify_legacy_exclusions` at analyzer
  read-time, idempotently.
- **2026-05-13 (also):** Staged rollout flag added. The sweep can be run
  in two stages via `--anchors skip` (pairs only) then `--anchors only`
  (anchors only), with resumability ensuring stage 2 only runs new cells.
  This lets the experimenter validate the Chinese open-weight slice and
  observe actual costs before committing the Claude anchor.
- **2026-05-15 (max\_tokens bug-fix + V4-pro routing change):**
  Discovered that `max_tokens=2048` (set in `configs/pivot_a.yaml`
  for all axes) was below the DeepSeek-recommended default for
  reasoning models. Per the public DeepSeek API docs, the default
  for `deepseek-reasoner` is 32K and the maximum is 64K; our 2048
  was 16× too small. On hardness-5 arithmetic the reasoning model
  consumed the entire 2048-token budget on `reasoning_content`
  (the CoT field), leaving zero tokens for the final `content`
  field. The runner recorded these as `empty_probe_answer` or
  `provider_empty_response` exclusions on DeepSeek V4-flash
  reasoning (~668/1000 cells on self-consistency) and V4-pro
  reasoning (~600/1000 cells on self-consistency, similar pattern
  across other axes).
  Fixes applied:
  (a) `src/agent_pathologies/client.py` now reads
  `reasoning_content` (DeepSeek-direct) or `reasoning` /
  `reasoning_details` (OpenRouter) as a fallback when `content` is
  empty, providing defense-in-depth recovery;
  (b) `scripts/retry_deepseek_empties.py` re-attempts each affected
  cell with `max_tokens=8192` on the same model+seed+sweep tuple,
  appending the new row to the existing JSONL (which remains
  append-only as an audit trail);
  (c) V4-pro reasoning is rerouted from
  OpenRouter/Novita to DeepSeek-direct for the retry pass — this
  exploits the 75\% V4-pro promotional discount and provider-side
  prompt caching, removes third-party serving variability, and
  places both members of the V4-pro pair on the model creator's
  first-party API (the same upstream-consistency principle that
  motivated the original pinning rule). The retry achieved a 100\%
  recovery rate on the first 41 cells tested.
  This was an experimental-configuration error in our setup, not a
  property of the DeepSeek model or its serving stack. The §6
  exclusion table reports post-retry effective counts (cells where
  every attempt for that cell failed); `dedupe_to_latest` in
  `metrics.py` ensures superseded rows are not double-counted.
  Discussion section 7.4 (originally planned as a "serving-stack
  signal" subsection on the empty-response asymmetry) is dropped.
- **2026-05-14 (resumability bug-fix):** Discovered and fixed a
  cell-key collision in the resumability mechanism. The original
  `cell_key()` hashed `(model, task_id, sweep_value, seed)`; for the
  DeepSeek V4-pro pair, both instruct and reasoning roles use the same
  model ID (`deepseek/deepseek-v4-pro`) and differ only in the runtime
  `reasoning_config` parameter, so the two roles' cells could collide.
  When one role completed a cell, the OTHER role's matching cell would
  be marked done in the resume set and not re-attempted. **The stored
  trajectories were unaffected — every row in the JSONL is internally
  correct, scored against the right answer, and labelled with the right
  `model_role`. The bug was purely in deciding which NEW cells to
  attempt.** The fix adds `model_role` to the key derivation and makes
  `existing_cell_keys` recompute keys from each stored row's fields
  (rather than reading `extra.cell_key`), so old rows are correctly
  classified under the new key formula. On the subsequent resume the
  previously-skipped V4-pro cells are re-attempted; the other six
  models are unaffected because their model IDs already differed across
  roles. We document this here because the change affects
  \emph{which} trajectories appear in the final dataset (more, not
  fewer; data quality unchanged), and reviewers should see the
  chronology.
