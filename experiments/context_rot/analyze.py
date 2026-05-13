"""Paired analysis of context_rot: paired McNemar on is_correct per
(family, kind, n_filler) cell, comparing instruct vs reasoning siblings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_pathologies.analysis.metrics import (
    accuracy_with_ci,
    exclusion_report,
    exploratory_families,
    filter_analyzable,
    load_jsonl,
    tag_exploratory,
)
from agent_pathologies.analysis.plots import plot_paired_bars, plot_accuracy_curve
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
    df["n_filler"] = df["sweep_value"].apply(lambda x: x["n_filler"])
    df["kind"] = df["sweep_value"].apply(lambda x: x["kind"])

    # --- marginal accuracy curves per (family, role, kind) ---
    rows = []
    for (family, role, kind, k), grp in df.groupby(
        ["model_family", "model_role", "kind", "n_filler"]
    ):
        stats = accuracy_with_ci(grp["is_correct"].tolist(), n_iters=args.bootstrap_iters)
        rows.append({
            "family": family, "role": role, "kind": kind, "n_filler": k,
            **stats,
        })
    marginal = pd.DataFrame(rows)
    print()
    print("=== marginal accuracy by (family, role, kind, n_filler) ===")
    print(marginal.to_string(index=False))

    # --- paired McNemar per (family, kind, n_filler) ---
    paired_rows = []
    pair_families = [f for f in df["model_family"].unique()
                     if {"instruct", "reasoning"}.issubset(
                         set(df[df["model_family"] == f]["model_role"].unique())
                     )]
    for family in pair_families:
        sub = df[df["model_family"] == family]
        for (kind, k), grp in sub.groupby(["kind", "n_filler"]):
            instr = grp[grp["model_role"] == "instruct"].set_index("task_id")["is_correct"]
            reas = grp[grp["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
            common = instr.index.intersection(reas.index)
            if len(common) < 5:
                continue
            a = instr.loc[common].astype(bool).tolist()
            b = reas.loc[common].astype(bool).tolist()
            res = paired_test(a, b)
            paired_rows.append({
                "family": family, "kind": kind, "n_filler": k,
                "n_paired": res.n,
                "acc_instruct": res.p_a, "acc_reasoning": res.p_b,
                "cohens_h": res.cohens_h, "p_value": res.p_value,
            })

    if paired_rows:
        q = benjamini_hochberg([r["p_value"] for r in paired_rows])
        for r, qv in zip(paired_rows, q):
            r["q_value_bh"] = qv
        paired_df = pd.DataFrame(paired_rows)
        explor = exploratory_families(df_all)
        if explor:
            paired_df["family"] = paired_df["family"].apply(
                lambda f: tag_exploratory(f, explor)
            )
        print()
        print("=== paired McNemar (instruct vs reasoning), context_rot ===")
        print(paired_df.to_string(index=False))
        if args.csv_out:
            Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
            paired_df.to_csv(args.csv_out, index=False)
            print(f"wrote {args.csv_out}")
    else:
        print("(no families with ≥5 paired tasks in any cell)")

    # --- plots ---
    out_dir = Path("data/plots")
    for kind in marginal["kind"].unique():
        sub = marginal[marginal["kind"] == kind]
        plot_accuracy_curve(
            sub,
            x_col="n_filler",
            hue_col="model_family",
            out=out_dir / f"context_rot_{kind}.png",
            title=f"Context rot — filler kind: {kind}",
        )

    if paired_rows:
        # Pick mid-sized filler for paired bars
        k_target = sorted(set(r["n_filler"] for r in paired_rows))[
            len(set(r["n_filler"] for r in paired_rows)) // 2
        ]
        for kind in sorted(set(r["kind"] for r in paired_rows)):
            bar_rows = []
            for (family, role), grp in df[(df["n_filler"] == k_target) & (df["kind"] == kind)].groupby(
                ["model_family", "model_role"]
            ):
                stats = accuracy_with_ci(grp["is_correct"].tolist())
                bar_rows.append({"model_family": family, "model_role": role, **stats})
            if bar_rows:
                plot_paired_bars(
                    pd.DataFrame(bar_rows),
                    pair_col="model_family",
                    group_col="model_role",
                    out=out_dir / f"context_rot_paired_{kind}_k{k_target}.png",
                    title=f"context_rot @ k={k_target}, filler={kind}",
                )
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/context_rot.jsonl")
    p.add_argument("--csv-out", default="data/context_rot_paired.csv")
    p.add_argument("--bootstrap-iters", type=int, default=10_000)
    main(p.parse_args())
