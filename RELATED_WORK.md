# Related Work — and where your framing has prior-work risk

**TL;DR.** A focused arxiv pass turned up significant prior work for all three
of the experiments you scoped. The original framing — "measure context rot,
sycophancy persistence, and self-consistency cleanly across models" — is
**mostly already published.** This document lists the closest prior works and
proposes pivots that keep the testbed you've built but reposition the
contribution.

Risk legend: 🔴 HIGH (your idea is already published), 🟡 MEDIUM (related but
your framing differs), 🟢 LOW (clear gap remains).

---

## 1. Context Rot — 🔴 HIGH overlap

| Paper | Claim | Overlap |
|---|---|---|
| [Laban et al., "LLMs Get Lost In Multi-Turn Conversation" (arXiv:2505.06120, May 2025)](https://arxiv.org/abs/2505.06120) | 200k+ simulated multi-turn conversations across six tasks → average **39% performance drop** vs single-turn. Decomposes into (small) aptitude loss + (large) unreliability increase. | 🔴 Covers the headline phenomenon you wanted to measure. Beating their 200k-conversation scale on the same finding is not realistic. |
| ["Drift No More? Context Equilibria in Multi-Turn LLM Interactions" (arXiv:2510.07777)](https://arxiv.org/html/2510.07777v1) | Frames multi-turn instability as a context equilibrium problem. | 🟡 Same phenomenon, theoretical lens. |
| ["Intent Mismatch Causes LLMs to Get Lost..." (arXiv:2602.07338)](https://arxiv.org/html/2602.07338v1) | Identifies intent mismatch as a primary driver of multi-turn degradation. | 🟡 Causal explanation work, separate from raw measurement. |
| ["Conversation Tree Architecture" (arXiv:2603.21278)](https://arxiv.org/html/2603.21278) | Coins "logical context poisoning" — progressive corruption from structural context mismanagement. | 🟡 Adjacent framing. |
| [NOLIMA (arXiv:2502.05167)](https://arxiv.org/pdf/2502.05167) | Long-context eval beyond literal matching. | 🟢 Adjacent — single-prompt, not multi-turn. |
| [MemoryAgentBench (arXiv:2507.05257)](https://arxiv.org/html/2507.05257v1) | Memory-mechanism evaluation in agents via incremental multi-turn input. | 🟡 Tests memory architectures, not vanilla LLM forgetting. |

**Verdict.** The vanilla "accuracy decays as filler turns pile up" curve is
covered. Your only path forward on context rot is either (a) a mechanistic
finding, or (b) a comparison axis nobody else ran (Chinese open weights,
reasoning vs non-reasoning).

---

## 2. Sycophancy Persistence — 🔴 HIGH overlap (most exposed of the three)

| Paper | Claim | Overlap |
|---|---|---|
| [Hong et al., "Measuring Sycophancy in Multi-turn Dialogues" / SYCON Bench (arXiv:2505.23840, EMNLP 2025)](https://arxiv.org/abs/2505.23840) | Introduces **Turn of Flip (ToF)** and **Number of Flip (NoF)** as metrics for sycophancy across multi-turn pressure; also reports that reasoning-optimized models generally resist sycophancy better than instruction-tuned variants. | 🔴 Literally your sycophancy hypothesis, with named metrics and a reasoning-vs-instruct result. Published at EMNLP 2025. |
| [Fanous et al., "SycEval: Evaluating LLM Sycophancy" (arXiv:2502.08177)](https://arxiv.org/html/2502.08177v4) | Reports that **"once a model yields to a user assertion, agreement-seeking behavior often persists across subsequent turns"** — verbatim your hypothesis. | 🔴 Direct statement of your sycophancy persistence finding. |
| ["Sycophancy under Pressure" (arXiv:2508.13743)](https://arxiv.org/html/2508.13743v1) | Adversarial dialogues for scientific QA sycophancy. | 🟡 Domain-specific; method differs. |
| ["Sycophancy Is Not One Thing: Causal Separation" (arXiv:2509.21305)](https://arxiv.org/html/2509.21305v1) | Decomposes sycophantic behaviors causally. | 🟡 Mechanistic angle, complementary. |
| ["Beacon: Single-Turn Diagnosis of Latent Sycophancy" (arXiv:2510.16727)](https://arxiv.org/html/2510.16727) | Single-turn detection of latent sycophantic tendencies. | 🟡 Different setting. |
| ["The Price of Agreement: LLM Sycophancy in Financial Apps" (arXiv:2604.24668)](https://arxiv.org/html/2604.24668) | Sycophancy in agentic financial applications. | 🟢 Domain-specific, not method. |
| [SycoEval-EM (arXiv:2601.16529)](https://arxiv.org/html/2601.16529) | Sycophancy benchmark in clinical encounters; reports DeepSeek-chat-v3.1 at **48% acquiescence rate**. | 🟡 Already evaluating DeepSeek. |
| ["Fragility of Moral Judgment" (arXiv:2603.05651)](https://arxiv.org/html/2603.05651) | Sycophancy perturbations; evaluates DeepSeek-V3 and Qwen2.5-72B. | 🟡 Already evaluating Chinese open-weight on sycophancy. |

**Verdict.** The most exposed of the three. SYCON and SycEval together cover
the metric, the protocol, and the qualitative finding. Pivot or drop.

---

## 3. Self-Consistency Drift — 🔴 HIGH overlap

| Paper | Claim | Overlap |
|---|---|---|
| [Atil et al., "Non-Determinism of 'Deterministic' LLM Settings" (arXiv:2408.04667)](https://arxiv.org/abs/2408.04667) | Five LLMs, eight tasks, 10 runs each. Up to **15% accuracy variance** and 70% best-to-worst gap even with deterministic config. | 🔴 Direct measurement of your hypothesis. |
| [He et al., "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference" (arXiv:2506.09501)](https://arxiv.org/pdf/2506.09501) | **Mechanistic explanation**: tiny logit gaps + BF16/FP16 numerical fluctuations. FP32 → near-deterministic; BF16 → significant variance. | 🔴 The mechanism is already published. |
| ["The Non-Determinism of Small LLMs" (arXiv:2509.09705)](https://arxiv.org/html/2509.09705v1) | Repeated MCQ trials on 2B–8B models. | 🟡 Different model-size band. |
| ["LLM Output Drift: Cross-Provider Validation" (arXiv:2511.07585)](https://arxiv.org/html/2511.07585) | Exactly the "different providers give different answers for the same model" finding. | 🔴 Cross-provider determinism already measured. |
| ["Measuring Determinism in LLMs for Code Review" (arXiv:2502.20747)](https://arxiv.org/pdf/2502.20747) | Determinism evaluation for code tasks. | 🟡 Domain-specific. |

**Verdict.** As a standalone finding, dead. Keep `self_consistency` *only* as
the noise floor for whichever pivot you land on — don't try to publish it.

---

## 4. Agent failure taxonomies — relevant context, low overlap

| Paper | Claim |
|---|---|
| ["Where LLM Agents Fail and How They Can Learn From Failures" (arXiv:2509.25370)](https://arxiv.org/abs/2509.25370) | AgentErrorTaxonomy + AgentErrorBench — annotated failure trajectories on ALFWorld/GAIA/WebShop. |
| [TRAJECT-Bench (arXiv:2510.04550)](https://arxiv.org/abs/2510.04550v1) | Trajectory-aware benchmark for agentic tool use; identifies "similar tool confusion" and "parameter-blind selection." |
| [AgenTracer (arXiv:2509.03312)](https://arxiv.org/pdf/2509.03312) | Failure localization in multi-agent systems. |

These are about *tool-using* agent failure, not multi-turn conversational
pathologies. Useful related-work neighbors, not direct competitors.

---

## What's still novel — your pivot options

The original framing is too crowded. Here are the angles where I see a real
gap, ranked by what I'd bet on as a paper:

### Pivot A — Reasoning vs non-reasoning split (RECOMMENDED)
**Run the testbed you've already built on reasoning *and* non-reasoning
siblings of the same family**: `deepseek-v4-pro` vs `deepseek-r1-0528`
*(exploratory — these two differ in base architecture/scale; the headline
rests on the Qwen pairs)*, `qwen3-235b-a22b-2507` vs
`qwen3-235b-a22b-thinking-2507`, `qwen3-30b-a3b-instruct-2507` vs
`qwen3-30b-a3b-thinking-2507`, with `anthropic/claude-opus-4.7` as anchor.

**Hypothesis.** Reasoning-trained models trade off pathology-resistance
differently than instruct-trained siblings. Either (a) reasoning models are
more robust because they re-derive answers, or (b) they are *less* robust
because they over-explain themselves into sycophantic reversals. Either
direction is publishable if the paper stays framed as a paired multi-axis
profile rather than a standalone sycophancy benchmark.

**Why this works:** the broad multi-axis, paired, no-LLM-judge comparison is
still a useful gap. However, sycophancy alone is no longer enough, because
SYCON already includes a reasoning-vs-instruction-tuned result. The strongest
claim is the combined trajectory-pathology profile across context rot,
sycophancy, and deterministic replay noise.

**Risk:** medium — the paper must not sell "reasoning vs instruct
sycophancy" as the main novelty. The novelty is the controlled, paired,
multi-axis pathology profile.

### Pivot B — Mechanistic interpretability on one phenomenon
Pick one (probably context rot), load Qwen3-32B locally via `transformers`,
and probe attention from the probe token back to the plant token across the
filler-count sweep. **Identify the attention pattern that decays in lockstep
with accuracy.** Bonus: causal intervention — patch the attention activations
back to their turn-0 values and see if accuracy recovers.

**Why this works:** none of the multi-turn-degradation papers in this list
do interpretability. This is the move from "we measured a thing" to "we
found the circuit responsible." Top-venue paper if the result is clean.

**Risk:** higher — open-ended ML research, may not pan out. Time budget:
3–4 weeks of GPU work after the behavioral curves are in.

### Pivot C — First systematic evaluation on Chinese open-weight stack
Frame as a measurement paper: replicate SYCON / SycEval / "LLMs Get Lost"
methodology on a model set nobody has run cleanly (Qwen 3.5, DeepSeek V4,
GLM 4.7, Kimi K2.6, MiniMax M2.7). Add reasoning-model coverage. Add
size-scaling rows.

**Why this works:** "first systematic eval on X" is a legitimate paper
shape, and the eval-on-Chinese-models gap is real. Workshop / regional
venue acceptance odds are high.

**Risk:** medium — reviewers may call it derivative. Compensate with breadth
and an open-source dashboard contribution.

### Pivot D — Drop sycophancy, focus on one novel axis of context rot
Examples: cross-turn fact contamination (does an injected wrong fact early
in a long conversation propagate into unrelated later answers?), or
adversarial filler (filler turns specifically designed to compete with
the planted fact for attention). The basic curve is published — but
specific perturbation typologies aren't.

**Risk:** medium — needs you to identify the specific perturbation
nobody has tried.

---

## Decision: Pivot A locked (2026-05-13)

After the lit pass, **Pivot A is committed** as the paper direction. The
testbed remains the same; the comparison axis is now within-family instruct
vs reasoning. Full hypotheses, analysis plan, and stopping rules are in
`PREREGISTRATION.md`.

The remaining pivots (B mechanistic, C systematic-eval-on-Chinese-stack,
D novel-perturbation) stay on the shelf as follow-up papers if a strong
effect appears in A.

## Why Pivot A

**Pick Pivot A (reasoning vs non-reasoning).** Reasons:
1. Reuses the testbed you've already built — zero engineering rework.
2. The model pairs exist on OpenRouter today (DeepSeek V4 vs R1, Qwen
   instruct vs thinking).
3. Whatever the result, it's publishable if framed as a multi-axis pathology
   profile, not as a standalone sycophancy benchmark.
4. If a striking effect shows up on one phenomenon, you have a natural
   path into Pivot B (mechanistic) on that phenomenon.

If you commit to Pivot A, the updated paper outline becomes:
> *Do reasoning-tuned LLMs exhibit different trajectory pathologies than
> their instruct-tuned siblings? A controlled study across the Qwen, DeepSeek,
> and Anthropic model families.*

The three experiments (`self_consistency`, `context_rot`, `sycophancy`)
become the three measurement axes within that paper. The story stays
cohesive. The novelty is the within-family comparison.

## Source list

- [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120)
- [Drift No More? Context Equilibria in Multi-Turn LLM Interactions](https://arxiv.org/html/2510.07777v1)
- [Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation](https://arxiv.org/html/2602.07338v1)
- [Conversation Tree Architecture](https://arxiv.org/html/2603.21278)
- [SYCON Bench — Measuring Sycophancy in Multi-turn Dialogues](https://arxiv.org/abs/2505.23840)
- [SycEval](https://arxiv.org/html/2502.08177v4)
- [Sycophancy under Pressure](https://arxiv.org/html/2508.13743v1)
- [Sycophancy Is Not One Thing: Causal Separation](https://arxiv.org/html/2509.21305v1)
- [Beacon — Single-Turn Diagnosis of Latent Sycophancy](https://arxiv.org/html/2510.16727)
- [SycoEval-EM — clinical sycophancy](https://arxiv.org/html/2601.16529)
- [The Fragility of Moral Judgment in LLMs](https://arxiv.org/html/2603.05651)
- [Non-Determinism of "Deterministic" LLM Settings](https://arxiv.org/abs/2408.04667)
- [Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/pdf/2506.09501)
- [The Non-Determinism of Small LLMs](https://arxiv.org/html/2509.09705v1)
- [LLM Output Drift: Cross-Provider Validation](https://arxiv.org/html/2511.07585)
- [NOLIMA — Long-Context Beyond Literal Matching](https://arxiv.org/pdf/2502.05167)
- [MemoryAgentBench](https://arxiv.org/html/2507.05257v1)
- [Where LLM Agents Fail and How They Can Learn From Failures](https://arxiv.org/abs/2509.25370)
- [TRAJECT-Bench](https://arxiv.org/abs/2510.04550v1)
- [AgenTracer](https://arxiv.org/pdf/2509.03312)
