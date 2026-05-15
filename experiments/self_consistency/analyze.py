"""Paired analysis of self_consistency: instruct vs reasoning per family,
using divergence-per-task and a paired Wilcoxon signed-rank test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from agent_pathologies.analysis.metrics import (
    answer_divergence,
    exclusion_report,
    exploratory_families,
    extracted_divergence,
    filter_analyzable,
    load_jsonl,
    tag_exploratory,
)
from agent_pathologies.analysis.stats import benjamini_hochberg, bootstrap_ci


def main(args: argparse.Namespace) -> None:
    df_all = load_jsonl(Path(args.path))
    if df_all.empty:
        print(f"no rows in {args.path}")
        return

    print("=== exclusions ===")
    ex = exclusion_report(df_all)
    print((ex.to_string(index=False)) if not ex.empty else "(none)")

    df = filter_analyzable(df_all)
    if df.empty:
        print("\n(no analyzable rows remain after exclusions — analyzer cannot proceed)")
        return

    # Per-task divergence per (family, role).
    # Primary divergence metric is on the EXTRACTED integer answer (what the
    # task actually scores). String-form divergence is also reported for
    # comparison — the analysis pass on stage-1 data found that string
    # divergence overstated the deepseek-v4-flash effect by ~2x because
    # chain-of-thought text varies in length even when the final integer
    # matches. See PREREGISTRATION amendment dated 2026-05-14.
    rows = []
    for (family, role, task_id), grp in df.groupby(["model_family", "model_role", "task_id"]):
        if len(grp) < 3:
            continue
        d_str = answer_divergence(grp["probe_answer"].tolist())
        d_int = extracted_divergence(grp["probe_answer"].tolist())
        acc = grp["is_correct"].mean()
        rows.append({
            "family": family, "role": role, "task_id": task_id,
            "divergence": d_int,            # primary: extracted integer
            "divergence_string": d_str,     # legacy/reference: raw string
            "accuracy": acc, "n": len(grp),
        })
    per_task = pd.DataFrame(rows)
    if per_task.empty:
        print("no per-task data")
        return

    print()
    print("=== mean (integer) divergence per family x role ===")
    summary = (
        per_task.groupby(["family", "role"])["divergence"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    print(summary.to_string(index=False))

    print()
    print("=== mean accuracy per family x role ===")
    acc_summary = (
        per_task.groupby(["family", "role"])["accuracy"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    print(acc_summary.to_string(index=False))

    # Paired Wilcoxon per family
    print()
    print("=== paired Wilcoxon (instruct vs reasoning), divergence ===")
    families = [f for f in per_task["family"].unique()
                if {"instruct", "reasoning"}.issubset(
                    set(per_task[per_task["family"] == f]["role"].unique())
                )]
    if not families:
        print("(no within-family pairs available — need both instruct and reasoning rows)")
        return

    def _paired_wilcoxon(metric_col: str):
        rs = []
        for family in families:
            sub = per_task[per_task["family"] == family]
            instr = sub[sub["role"] == "instruct"].set_index("task_id")[metric_col]
            reas = sub[sub["role"] == "reasoning"].set_index("task_id")[metric_col]
            common = instr.index.intersection(reas.index)
            if len(common) < 5:
                continue
            a, b = instr.loc[common].values, reas.loc[common].values
            if (a - b == 0).all():
                p = 1.0
            else:
                p = float(wilcoxon(a, b, zero_method="wilcox",
                                   alternative="two-sided").pvalue)
            ci_lo, ci_hi = bootstrap_ci((a - b).tolist(),
                                        n_iters=args.bootstrap_iters)
            rs.append({
                "family": family, "metric": metric_col,
                "n_paired": int(len(common)),
                "instr_mean": float(a.mean()),
                "reas_mean": float(b.mean()),
                "delta_mean": float((a - b).mean()),
                "delta_ci_lo": ci_lo, "delta_ci_hi": ci_hi,
                "wilcoxon_p": p,
            })
        return rs

    # Co-primary tests (PREREGISTRATION amendment 2026-05-14): both
    # divergence and accuracy are tested with paired Wilcoxon. Accuracy
    # catches the Qwen mode-collapse pattern (consistently wrong same answer)
    # that pure divergence misses.
    paired_rows = _paired_wilcoxon("divergence") + _paired_wilcoxon("accuracy")

    if not paired_rows:
        print("(no families with ≥5 paired tasks)")
        return

    # BH-correct within each metric family separately.
    div_rows = [r for r in paired_rows if r["metric"] == "divergence"]
    acc_rows = [r for r in paired_rows if r["metric"] == "accuracy"]
    if div_rows:
        qs = benjamini_hochberg([r["wilcoxon_p"] for r in div_rows])
        for r, q in zip(div_rows, qs):
            r["q_value_bh"] = q
    if acc_rows:
        qs = benjamini_hochberg([r["wilcoxon_p"] for r in acc_rows])
        for r, q in zip(acc_rows, qs):
            r["q_value_bh"] = q

    out_df = pd.DataFrame(div_rows + acc_rows)
    explor = exploratory_families(df_all)
    if explor:
        out_df["family"] = out_df["family"].apply(
            lambda f: tag_exploratory(f, explor)
        )
    print(out_df.to_string(index=False))

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.csv_out, index=False)
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/self_consistency.jsonl")
    p.add_argument("--csv-out", default="data/self_consistency_paired.csv")
    p.add_argument("--bootstrap-iters", type=int, default=10_000)
    main(p.parse_args())
