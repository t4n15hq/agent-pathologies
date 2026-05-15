# Related Work

This document situates `agent-pathologies` against the three literatures
it most directly speaks to — multi-turn conversational degradation,
sycophancy, and inference non-determinism — plus a fourth (agent failure
taxonomies) that frames the broader context. For each line of work we
describe what the cited paper establishes, what its design choices
constrain, and how the present study either re-uses or extends it.

The motivating observation is that each of these three pathologies has
been studied in isolation, often on heterogeneous model panels that
mix instruction-tuned and reasoning-tuned variants across families and
scales. The gap we target is a *controlled paired comparison*:
within-family instruct↔reasoning siblings tested on all three axes in
the same experimental harness, with frozen hypotheses, exclusion
classes, and effect-size thresholds. The point is not to re-measure
any single axis at higher fidelity than the works below — it is to
produce a coherent multi-axis profile of the reasoning-mode
intervention that the panel-average results in the prior literature
cannot disentangle.

---

## 1. Multi-turn conversational degradation ("context rot")

**Laban et al., *LLMs Get Lost in Multi-Turn Conversation*
([arXiv:2505.06120](https://arxiv.org/abs/2505.06120))** is the
canonical reference. The authors run roughly 200,000 simulated
multi-turn conversations across six tasks and report an average 39%
performance drop relative to single-turn execution, decomposed into a
small aptitude loss and a larger unreliability increase. The paper
establishes that the degradation is real and large at panel scale; it
does not isolate the reasoning-mode contrast, and its model panel is
heterogeneous (different families, different post-training).

**Drift No More? Context Equilibria in Multi-Turn LLM Interactions
([arXiv:2510.07777](https://arxiv.org/html/2510.07777v1))** frames
multi-turn instability as a context-equilibrium dynamical problem,
giving a theoretical lens for the same phenomenon. **Intent Mismatch
Causes LLMs to Get Lost in Multi-Turn Conversation
([arXiv:2602.07338](https://arxiv.org/html/2602.07338v1))** isolates
intent mismatch as one mechanistic driver of the degradation. The
*Conversation Tree Architecture* paper
([arXiv:2603.21278](https://arxiv.org/html/2603.21278)) introduces the
"logical context poisoning" framing — progressive corruption of the
conversation state from structural mismanagement. **NOLIMA
([arXiv:2502.05167](https://arxiv.org/pdf/2502.05167))** extends
long-context evaluation past literal substring matching. **MemoryAgentBench
([arXiv:2507.05257](https://arxiv.org/html/2507.05257v1))** evaluates
memory mechanisms in agentic systems via incremental multi-turn input;
it studies architectural augmentations rather than vanilla LLM
forgetting.

**How this study relates.** Our context-rot axis adopts the
single-task, controlled-filler protocol — a 20-update variable-tracking
task probed after $k \in \{0, 2, 5, 10, 20, 40\}$ filler turn-pairs of
four kinds (irrelevant, topically related, token-matched,
collapsed-same-token-mass). We do not attempt to compete with Laban et
al. at panel scale; we instead trade breadth for the within-pair
reasoning-mode contrast they cannot deliver. Specifically, we identify
that the DeepSeek instruct siblings already saturate this task at
~$1.000$ accuracy across every filler depth — a ceiling effect that
suppresses any within-pair reasoning effect by construction. Reporting
this null honestly is itself a contribution, because the prior
literature's panel-average effect can be interpreted as a uniform
property of multi-turn LLM behavior when it is in fact a property of
the weakest models on the panel.

---

## 2. Sycophancy

**Hong et al., *Measuring Sycophancy in Multi-turn Dialogues* / SYCON
Bench ([arXiv:2505.23840](https://arxiv.org/abs/2505.23840), EMNLP
2025)** is the closest existing work to our sycophancy axis. SYCON
introduces *Turn of Flip (ToF)* and *Number of Flip (NoF)* as
metrics, reports that reasoning-optimized models in their panel
resist sycophancy better than instruction-tuned ones, and frames the
contrast at the cross-family level. **Fanous et al., *SycEval*
([arXiv:2502.08177](https://arxiv.org/html/2502.08177v4))** reports
that once a model yields to an incorrect user assertion the
agreement-seeking behavior often persists across subsequent turns —
the same observation we operationalize as the post-pushback gap sweep.
*Sycophancy under Pressure*
([arXiv:2508.13743](https://arxiv.org/html/2508.13743v1)) studies
adversarial dialogues in scientific QA. *Sycophancy Is Not One Thing*
([arXiv:2509.21305](https://arxiv.org/html/2509.21305v1)) decomposes
sycophantic behavior causally. *Beacon*
([arXiv:2510.16727](https://arxiv.org/html/2510.16727)) attempts
single-turn diagnosis of latent sycophantic tendencies, distinct from
the multi-turn pushback setup. *SycoEval-EM*
([arXiv:2601.16529](https://arxiv.org/html/2601.16529)) reports
DeepSeek-Chat acquiescence rates near 48% on clinical encounter
benchmarks. *The Fragility of Moral Judgment*
([arXiv:2603.05651](https://arxiv.org/html/2603.05651)) evaluates
DeepSeek-V3 and Qwen2.5 on moral-reasoning sycophancy perturbations.

**How this study relates.** SYCON's headline finding — that
reasoning-optimized variants resist sycophancy better — is the
closest direct prediction in the prior literature for our sycophancy
axis. We adopt the wrong / correct / neutral pushback design and the
post-pushback gap sweep, scoring on integer-extracted answers without
an LLM judge. The contribution is methodological: SYCON aggregates
across a heterogeneous panel where the reasoning effect is confounded
with model family, scale, and serving stack. Our within-MODEL pairs
(DeepSeek V4-pro toggled between reasoning on and off; V4-flash
toggled via its legacy aliases) isolate the reasoning-mode contrast
on identical base weights. The empirical result is more
heterogeneous than SYCON's panel-average suggests: we find a clean
within-MODEL effect on V4-flash, a null at the preregistered
threshold on V4-pro, and large but capability-confounded effects on
the Qwen cross-SKU pairs. The right read of our paper alongside
SYCON's is that the panel-average direction holds in expectation but
the within-pair effect is not uniform across reasoning-enabled
variants, which is a practitioner-relevant refinement.

---

## 3. Inference non-determinism ("self-consistency drift")

**Atil et al., *Non-Determinism of "Deterministic" LLM Settings*
([arXiv:2408.04667](https://arxiv.org/abs/2408.04667))** measures up
to 15% accuracy variance and up to 70% best-to-worst spread across
ten runs of five LLMs on eight tasks, even with deterministic
sampling configuration. **He et al., *Understanding and Mitigating
Numerical Sources of Nondeterminism in LLM Inference*
([arXiv:2506.09501](https://arxiv.org/pdf/2506.09501))** provides the
mechanistic explanation — tiny logit gaps amplified by BF16/FP16
numerical noise — and shows that FP32 inference recovers near-determinism.
*The Non-Determinism of Small LLMs*
([arXiv:2509.09705](https://arxiv.org/html/2509.09705v1)) reports
repeated-trial variance specifically in the 2B–8B parameter band.
*LLM Output Drift: Cross-Provider Validation*
([arXiv:2511.07585](https://arxiv.org/html/2511.07585)) measures the
cross-provider variance for the same nominal model. *Measuring
Determinism in LLMs for Code Review*
([arXiv:2502.20747](https://arxiv.org/pdf/2502.20747)) evaluates
determinism on code tasks.

**How this study relates.** As a standalone phenomenon, inference
non-determinism is comprehensively measured by the works above; we
make no novel claim about the measurement. We re-use the protocol —
identical replays at $T=0$ scored by integer extraction — as a noise
floor against which the within-pair reasoning effect must register.
The substantive finding on our axis is that the reasoning siblings
drive integer-extracted answer divergence to near zero on the
DeepSeek pairs (from $0.422$ to $0.010$ on V4-flash; from $0.554$ to
$0.003$ on V4-pro) while also producing the largest accuracy gains in
the paper. We frame this as joint evidence that the reasoning mode
trades off the He et al. numerical-noise sensitivity for a more
stable conditional answer distribution, not as an independent
contribution to the determinism literature.

We additionally document one phenomenon the cited works do not: the
Qwen instruct SKUs mode-collapse on hardness-5 arithmetic to a
single placeholder integer (most often "1234"), giving low
divergence with zero accuracy. The accuracy paired test was
preregistered as a co-primary measure specifically to catch this
failure mode, which a divergence-only protocol would record as a
false null. We document this in §6.1 of the paper and use it as the
empirical motivation for reporting accuracy and divergence
separately on this axis.

---

## 4. Agent failure taxonomies

Three recent works characterize agent-failure modes at a higher level
of abstraction: **Where LLM Agents Fail and How They Can Learn From
Failures ([arXiv:2509.25370](https://arxiv.org/abs/2509.25370))**
introduces AgentErrorTaxonomy and AgentErrorBench, annotating failure
trajectories on ALFWorld, GAIA, and WebShop. **TRAJECT-Bench
([arXiv:2510.04550](https://arxiv.org/abs/2510.04550v1))** identifies
"similar tool confusion" and "parameter-blind selection" as recurring
failure modes in tool-using agents. **AgenTracer
([arXiv:2509.03312](https://arxiv.org/pdf/2509.03312))** localizes
failures in multi-agent systems. These taxonomies cover tool-using and
multi-agent contexts that are out of scope for our pure
single-LLM, multi-turn-conversation setting, but they place the
present study in the broader landscape of LLM-as-component reliability
work.

---

## 5. Reasoning-vs-instruct comparisons as a methodological axis

The closest work to our specific design (within-family, paired,
controlled) is SYCON's reasoning-vs-instruct sub-analysis on the
sycophancy axis. SYCON does not extend this contrast across
self-consistency or context rot, and the underlying pairs are
cross-family rather than within-MODEL. None of the works in §§1–3
above isolates the reasoning-mode intervention in a paired design with
frozen analysis plan across three pathology axes. The contribution of
the present paper is the controlled multi-axis profile rather than any
single-axis measurement: combining self-consistency, sycophancy, and
context rot in one experiment makes it possible to distinguish a
"reasoning helps everywhere" reading (which the data refutes) from a
"reasoning helps selectively" reading (which the data supports).

---

## 6. What this paper contributes that the prior work does not

Three claims:

1. **The first within-MODEL reasoning-toggle comparison.** Our DeepSeek
   pairs hold base weights, scale, and serving host identical across
   instruct and reasoning siblings; the only varying parameter is the
   runtime reasoning toggle. The prior literature's reasoning-vs-instruct
   comparisons (SYCON, SycEval, and most heterogeneous-panel work)
   compare across families or post-training runs, conflating the
   intervention with confounders we hold constant.

2. **A multi-axis pathology profile under a single preregistered
   analysis plan.** Hypotheses, primary metrics, effect-size thresholds
   (Cohen's $h \geq 0.20$; $|\text{DiD gain}| \geq 0.10$), BH
   multiple-comparisons correction at FDR$=0.05$, and the six
   exclusion classes were committed before any real-model data was
   collected. Every amendment is dated in `PREREGISTRATION.md`. The
   cross-axis pattern is the headline finding.

3. **Honest accounting of where the within-pair effect is heterogeneous
   and where it is ceiling-bound.** We report and label V4-pro
   sycophancy as a within-MODEL null at the preregistered threshold,
   even though the panel-average direction in SYCON is positive; and
   we report the DeepSeek context-rot result as a ceiling-bound null
   rather than a reverse finding. The substantive practitioner
   takeaway — reasoning enablement is a selective intervention, not a
   uniform robustness lever — depends on these honest nulls being
   reported, not hidden.
