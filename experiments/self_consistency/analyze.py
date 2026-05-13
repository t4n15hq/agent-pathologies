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
    filter_analyzable,
    load_jsonl,
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

    # Per-task divergence per (family, role)
    rows = []
    for (family, role, task_id), grp in df.groupby(["model_family", "model_role", "task_id"]):
        if len(grp) < 3:
            continue
        d = answer_divergence(grp["probe_answer"].tolist())
        acc = grp["is_correct"].mean()
        rows.append({
            "family": family, "role": role, "task_id": task_id,
            "divergence": d, "accuracy": acc, "n": len(grp),
        })
    per_task = pd.DataFrame(rows)
    if per_task.empty:
        print("no per-task data")
        return

    print()
    print("=== mean divergence per family x role ===")
    summary = (
        per_task.groupby(["family", "role"])["divergence"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    print(summary.to_string(index=False))

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

    paired_rows = []
    for family in families:
        sub = per_task[per_task["family"] == family]
        instr = sub[sub["role"] == "instruct"].set_index("task_id")["divergence"]
        reas = sub[sub["role"] == "reasoning"].set_index("task_id")["divergence"]
        common = instr.index.intersection(reas.index)
        if len(common) < 5:
            continue
        a, b = instr.loc[common].values, reas.loc[common].values
        # Skip if all differences are zero (no variability)
        if (a - b == 0).all():
            stat, p = float("nan"), 1.0
        else:
            res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            stat, p = float(res.statistic), float(res.pvalue)
        ci_lo_diff, ci_hi_diff = bootstrap_ci((a - b).tolist(), n_iters=args.bootstrap_iters)
        paired_rows.append({
            "family": family,
            "n_paired": int(len(common)),
            "div_instruct": float(a.mean()),
            "div_reasoning": float(b.mean()),
            "delta_mean": float((a - b).mean()),
            "delta_ci_lo": ci_lo_diff,
            "delta_ci_hi": ci_hi_diff,
            "wilcoxon_p": p,
        })

    if not paired_rows:
        print("(no families with ≥5 paired tasks)")
        return

    q = benjamini_hochberg([r["wilcoxon_p"] for r in paired_rows])
    for r, qv in zip(paired_rows, q):
        r["q_value_bh"] = qv

    out_df = pd.DataFrame(paired_rows)
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
