"""Paired analysis of sycophancy: paired McNemar on accuracy at re-probe,
per (family, condition, post_gap). Reports within-family difference between
wrong/correct/neutral pushback conditions to isolate true sycophancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_pathologies.analysis.metrics import (
    accuracy_with_ci,
    exclusion_report,
    filter_analyzable,
    load_jsonl,
)
from agent_pathologies.analysis.plots import plot_accuracy_curve, plot_paired_bars
from agent_pathologies.analysis.stats import benjamini_hochberg, paired_test


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
    df["post_gap"] = df["sweep_value"].apply(lambda x: x["post_gap"])
    df["condition"] = df["sweep_value"].apply(lambda x: x["condition"])

    # --- marginal accuracy ---
    rows = []
    for (family, role, condition, gap), grp in df.groupby(
        ["model_family", "model_role", "condition", "post_gap"]
    ):
        stats = accuracy_with_ci(grp["is_correct"].tolist(), n_iters=args.bootstrap_iters)
        rows.append({
            "family": family, "role": role, "condition": condition,
            "post_gap": gap, **stats,
        })
    marginal = pd.DataFrame(rows)
    print()
    print("=== marginal accuracy at re-probe, by (family, role, condition, gap) ===")
    print(marginal.to_string(index=False))

    # --- paired McNemar per (family, condition, gap) ---
    paired_rows = []
    pair_families = [f for f in df["model_family"].unique()
                     if {"instruct", "reasoning"}.issubset(
                         set(df[df["model_family"] == f]["model_role"].unique())
                     )]
    for family in pair_families:
        sub = df[df["model_family"] == family]
        for (condition, gap), grp in sub.groupby(["condition", "post_gap"]):
            instr = grp[grp["model_role"] == "instruct"].set_index("task_id")["is_correct"]
            reas = grp[grp["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
            common = instr.index.intersection(reas.index)
            if len(common) < 5:
                continue
            a = instr.loc[common].astype(bool).tolist()
            b = reas.loc[common].astype(bool).tolist()
            res = paired_test(a, b)
            paired_rows.append({
                "family": family, "condition": condition, "post_gap": gap,
                "n_paired": res.n,
                "acc_instruct": res.p_a, "acc_reasoning": res.p_b,
                "cohens_h": res.cohens_h, "p_value": res.p_value,
            })

    if paired_rows:
        q = benjamini_hochberg([r["p_value"] for r in paired_rows])
        for r, qv in zip(paired_rows, q):
            r["q_value_bh"] = qv
        paired_df = pd.DataFrame(paired_rows)
        print()
        print("=== paired McNemar (instruct vs reasoning), sycophancy ===")
        print(paired_df.to_string(index=False))
        if args.csv_out:
            Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
            paired_df.to_csv(args.csv_out, index=False)
            print(f"wrote {args.csv_out}")

    # --- condition-delta within-model (sycophancy signature) ---
    cond_rows = []
    for (family, role, gap), grp in df.groupby(["model_family", "model_role", "post_gap"]):
        accs = {}
        for cond in ["wrong", "correct", "neutral"]:
            sub = grp[grp["condition"] == cond]
            if len(sub) > 0:
                accs[cond] = sub["is_correct"].mean()
        if {"wrong", "correct"}.issubset(accs):
            cond_rows.append({
                "family": family, "role": role, "post_gap": gap,
                "acc_wrong": accs.get("wrong"),
                "acc_correct": accs.get("correct"),
                "acc_neutral": accs.get("neutral"),
                "syc_signature": accs["correct"] - accs["wrong"],
            })
    if cond_rows:
        cond_df = pd.DataFrame(cond_rows)
        print()
        print("=== within-model condition deltas (sycophancy signature = correct − wrong) ===")
        print(cond_df.to_string(index=False))

    # --- plots ---
    out_dir = Path("data/plots")
    for cond in marginal["condition"].unique():
        sub = marginal[marginal["condition"] == cond]
        plot_accuracy_curve(
            sub,
            x_col="post_gap",
            hue_col="model_family",
            out=out_dir / f"sycophancy_{cond}.png",
            title=f"Sycophancy — pushback={cond}",
        )

    if paired_rows:
        # Headline plot: gap=0, condition=wrong (max effect cell)
        bar_rows = []
        for (family, role), grp in df[
            (df["post_gap"] == 0) & (df["condition"] == "wrong")
        ].groupby(["model_family", "model_role"]):
            stats = accuracy_with_ci(grp["is_correct"].tolist())
            bar_rows.append({"model_family": family, "model_role": role, **stats})
        if bar_rows:
            plot_paired_bars(
                pd.DataFrame(bar_rows),
                pair_col="model_family",
                group_col="model_role",
                out=out_dir / "sycophancy_paired_wrong_gap0.png",
                title="Sycophancy @ wrong pushback, gap=0",
            )
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/sycophancy.jsonl")
    p.add_argument("--csv-out", default="data/sycophancy_paired.csv")
    p.add_argument("--bootstrap-iters", type=int, default=10_000)
    main(p.parse_args())
