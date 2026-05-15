# paper/

LaTeX source for the preprint *Reasoning Enablement as a Selective
Multi-Turn Pathology Intervention: A Preregistered Within-Family Study
of Four Chinese Open-Weight Model Pairs*.

The paper is structurally complete and compiles to `main.pdf` via
`tectonic main.tex` (single-binary modern LaTeX engine) or any standard
distribution that ships `pdflatex` + `bibtex`.

## Layout

- `main.tex` — entry point. Loads each numbered section via `\input{}`.
- `abstract.tex` — abstract.
- `01_introduction.tex` — research question, three pathology axes, the
  selective-intervention thesis.
- `02_related_work.tex` — positions the paper against the multi-turn,
  sycophancy, and inference-non-determinism literatures.
- `03_methodology.tex` — paired within-family design, exclusion classes
  with the 2026-05-15 amendment text incorporated, resumable runner.
- `04_hypotheses_and_analysis_plan.tex` — preregistered hypotheses,
  primary metrics, effect-size thresholds, BH correction, falsification
  criteria.
- `05_experimental_setup.tex` — model pairs and pinning, V4-pro routing
  amendment subsection (§5.2), `max_tokens` amendment subsection (§5.3),
  semantic-dedup subsection (§5.4), statistical pipeline.
- `06_results.tex` — full results, seven tables filled from the final
  deduped data: pairs/exclusions, self-consistency accuracy and
  divergence, sycophancy per-cell and per-gap DiD (in the
  reasoning-gain convention), context-rot per-cell and pooled-cell,
  cross-axis pass/fail summary. Includes the Qwen mode-collapse figure
  reference.
- `07_discussion.tex` — five subsections: selective pattern,
  capability/sycophancy disentanglement, within-MODEL vs cross-SKU,
  SYCON comparison, practical implication.
- `08_limitations.tex` — including the candid disclosure of the
  `max_tokens` configuration error and its recovery.
- `09_conclusion.tex` — two-paragraph close.
- `10_reproducibility.tex` — repository URL, run dates, trajectory
  totals, real cost breakdown ($222.71 across OR and DS-direct),
  SHA-256 hashes of the cleaned JSONLs.
- `references.bib` — full bibliography, verified against arXiv listings.
- `figures/` — eight publication-quality figures rendered from the
  final deduped dataset (PDF + PNG). `fig1_headline_forest` is the
  cross-axis forest plot; `fig2_selfconsistency_paired` is the
  self-consistency paired bars; `fig3_sycophancy_did` is the
  sycophancy DiD in the reasoning-gain convention with the symmetric
  $\pm 0.10$ threshold lines; `fig4_sycophancy_conditions` is the
  condition-accuracy heatmap; `fig5_contextrot_curves` is the
  irrelevant-filler decay (main); `fig5b_contextrot_full` is the
  four-kind decay (supplementary); `fig6_qwen_mode_collapse` is the
  Qwen instruct top-5 answer-frequency bars; `fig7_cross_axis_summary`
  is the signed-effect heatmap.

## Build

```bash
tectonic main.tex                # produces main.pdf
```

or, with a TeX Live / MacTeX installation:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The build is reproducible: bibliography keys are stable, all
references resolve, the figures are version-controlled PDFs.

## Notes

- The paper is a single-author independent-researcher preprint
  targeting arXiv cs.CL primary.
- All numeric content in the tables and prose is derived from the
  `data/*_clean.jsonl` files via the analyzer pipeline in
  `experiments/<axis>/analyze.py` and `scripts/pair_analysis.py`.
  Re-running those scripts on the same data reproduces every cell.
