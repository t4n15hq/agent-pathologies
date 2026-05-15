# paper/

LaTeX source for the Pivot A paper.

## Layout

- `main.tex` — entry point. `\input{}`s each numbered section.
- `abstract.tex` — full abstract draft with `[NUM]` placeholders.
- `01_introduction.tex` — drafted (~1.5 pp).
- `02_related_work.tex` — drafted (~1 pp).
- `03_methodology.tex` — drafted (~2.5 pp).
- `04_hypotheses_and_analysis_plan.tex` — drafted (~1 pp), reflects the
  amendment log through 2026-05-14.
- `05_experimental_setup.tex` — drafted (~0.5–1 pp), with the model
  and hyperparameter tables.
- `06_results.tex` — **placeholder** stub. Pending data collection.
- `07_discussion.tex` — **placeholder** stub.
- `08_limitations.tex` — initial draft (~0.5 pp).
- `09_conclusion.tex` — **placeholder** stub.
- `10_reproducibility.tex` — drafted reproducibility appendix (commit
  hash, configs, run dates, regen commands, file hashes — all with
  `[NUM]` / `[GIT SHA]` placeholders to fill at release time).
- `references.bib` — bibliography, populated from `RELATED_WORK.md`.
  Author lists are placeholders; verify against arXiv before submission.
- `figures/` — empty.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a standard LaTeX distribution with `natbib`, `booktabs`,
`microtype`, `tabularx`, `hyperref`.

## TODO

- Fill `06_results.tex` once `data/*_clean.jsonl` is finalised. The
  comment block in that file lists the four subsections to draft.
- Fill `07_discussion.tex` and `09_conclusion.tex` after results are in.
- Replace every `\NUM` (`[NUM]`) placeholder in `abstract.tex` and
  `01_introduction.tex` with the actual headline numbers.
- Generate figures into `figures/` from
  `experiments/<axis>/analyze.py` once the sweep completes.
- Re-verify `references.bib` author lists from arXiv abstracts before
  submission (entries currently use shorthand `Author, and others` for
  the second-author position where the lead author was clear from
  `RELATED_WORK.md` and the rest were truncated).

## Notes for the next session

- The abstract claims ``\NUM\ on self-consistency / wrong-pushback
  resistance / context rot, with the context-rot direction depending on
  whether the instruct baseline already sits near ceiling.'' The
  introduction commits to the same narrative shape. If the data does
  not bear this out, rewrite both before reading the results section.
- The paper is written as a single-author independent-researcher
  preprint (arXiv cs.CL primary). No venue style file is included.
