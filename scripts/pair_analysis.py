"""One-shot cross-experiment summary. Loads all three JSONL outputs,
computes the preregistered headline paired numbers per family, prints +
writes CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from agent_pathologies.analysis.metrics import (
    answer_divergence,
    filter_analyzable,
    load_jsonl,
)
from agent_pathologies.analysis.stats import paired_did_bootstrap, paired_test


def accuracy_headline_rows(df: pd.DataFrame, experiment: str, cell_filter):
    rows = []
    df = df[df["experiment"] == experiment]
    pair_families = [f for f in df["model_family"].unique()
                     if {"instruct", "reasoning"}.issubset(
                         set(df[df["model_family"] == f]["model_role"].unique())
                     )]
    for family in pair_families:
        sub = cell_filter(df[df["model_family"] == family])
        instr = sub[sub["model_role"] == "instruct"].set_index("task_id")["is_correct"]
        reas = sub[sub["model_role"] == "reasoning"].set_index("task_id")["is_correct"]
        common = instr.index.intersection(reas.index)
        if len(common) < 5:
            continue
        a = instr.loc[common].astype(bool).tolist()
        b = reas.loc[common].astype(bool).tolist()
        res = paired_test(a, b)
        rows.append({
            "experiment": experiment,
            "family": family,
            "metric": "accuracy",
            "n_paired": res.n,
            "value_instruct": res.p_a,
            "value_reasoning": res.p_b,
            "delta_instruct_minus_reasoning": res.p_a - res.p_b,
            "cohens_h": res.cohens_h,
            "p_value": res.p_value,
        })
    return rows


def self_consistency_rows(df: pd.DataFrame):
    rows = []
    df = df[df["experiment"] == "self_consistency"]
    if df.empty:
        return rows

    per_task_rows = []
    for (family, role, task_id), grp in df.groupby(["model_family", "model_role", "task_id"]):
        if len(grp) < 3:
            continue
        per_task_rows.append({
            "family": family,
            "role": role,
            "task_id": task_id,
            "divergence": answer_divergence(grp["probe_answer"].tolist()),
        })
    per_task = pd.DataFrame(per_task_rows)
    if per_task.empty:
        return rows

    pair_families = [f for f in per_task["family"].unique()
                     if {"instruct", "reasoning"}.issubset(
                         set(per_task[per_task["family"] == f]["role"].unique())
                     )]
    for family in pair_families:
        sub = per_task[per_task["family"] == family]
        instr = sub[sub["role"] == "instruct"].set_index("task_id")["divergence"]
        reas = sub[sub["role"] == "reasoning"].set_index("task_id")["divergence"]
        common = instr.index.intersection(reas.index)
        if len(common) < 5:
            continue
        a, b = instr.loc[common], reas.loc[common]
        diff = a.values - b.values
        if (diff == 0).all():
            p_value = 1.0
        else:
            p_value = float(wilcoxon(a.values, b.values, zero_method="wilcox").pvalue)
        rows.append({
            "experiment": "self_consistency",
            "family": family,
            "metric": "answer_divergence",
            "n_paired": int(len(common)),
            "value_instruct": float(a.mean()),
            "value_reasoning": float(b.mean()),
            "delta_instruct_minus_reasoning": float(diff.mean()),
            "cohens_h": float("nan"),
            "p_value": p_value,
        })
    return rows


def sycophancy_did_rows(df: pd.DataFrame, *, n_iters: int = 5000):
    """Per-family paired DiD: reasoning(c−w) − instruct(c−w), pooled across
    post-pushback gaps for a single headline number per family."""
    rows = []
    syc = df[df["experiment"] == "sycophancy"].copy()
    if syc.empty:
        return rows
    syc["condition"] = syc["sweep_value"].apply(lambda x: x["condition"])
    pair_families = [f for f in syc["model_family"].unique()
                     if {"instruct", "reasoning"}.issubset(
                         set(syc[syc["model_family"] == f]["model_role"].unique())
                     )]
    for family in pair_families:
        sub = syc[syc["model_family"] == family]
        try:
            ic = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "correct")
                     ].set_index("task_id")["is_correct"]
            iw = sub[(sub["model_role"] == "instruct") & (sub["condition"] == "wrong")
                     ].set_index("task_id")["is_correct"]
            rc = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "correct")
                     ].set_index("task_id")["is_correct"]
            rw = sub[(sub["model_role"] == "reasoning") & (sub["condition"] == "wrong")
                     ].set_index("task_id")["is_correct"]
        except KeyError:
            continue
        # Multiple rows per task_id (different gaps) — drop duplicates keeping mean.
        for s in (ic, iw, rc, rw):
            pass  # series may have repeats; mean below smooths over gaps.
        ic = ic.groupby(level=0).mean()
        iw = iw.groupby(level=0).mean()
        rc = rc.groupby(level=0).mean()
        rw = rw.groupby(level=0).mean()
        common = ic.index.intersection(iw.index).intersection(
            rc.index).intersection(rw.index)
        if len(common) < 5:
            continue
        res = paired_did_bootstrap(
            instr_correct=ic.loc[common].astype(float).tolist(),
            instr_wrong=iw.loc[common].astype(float).tolist(),
            reas_correct=rc.loc[common].astype(float).tolist(),
            reas_wrong=rw.loc[common].astype(float).tolist(),
            n_iters=n_iters,
        )
        rows.append({
            "experiment": "sycophancy",
            "family": family,
            "metric": "did",
            "n_paired": res.n_paired,
            "value_instruct": res.instr_gap,
            "value_reasoning": res.reas_gap,
            "delta_instruct_minus_reasoning": -res.did,  # sign-aligned with other rows
            "cohens_h": float("nan"),
            "p_value": res.bootstrap_p,
        })
    return rows


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    frames = []
    for name in ("self_consistency", "context_rot", "sycophancy"):
        path = data_dir / f"{name}.jsonl"
        if path.exists():
            f = load_jsonl(path)
            frames.append(filter_analyzable(f))
    if not frames:
        print("no data found in data_dir")
        return
    df = pd.concat(frames, ignore_index=True)

    rows: list[dict] = []
    rows += self_consistency_rows(df)

    # context_rot: headline cell is largest filler count, irrelevant
    def ctx_cell(d):
        d = d.copy()
        d["n_filler"] = d["sweep_value"].apply(lambda x: x["n_filler"])
        d["kind"] = d["sweep_value"].apply(lambda x: x["kind"])
        max_k = d["n_filler"].max()
        return d[(d["kind"] == "irrelevant") & (d["n_filler"] == max_k)]
    rows += accuracy_headline_rows(df, "context_rot", ctx_cell)

    # sycophancy: headline cell is condition=wrong, gap=0
    def syc_cell(d):
        d = d.copy()
        d["condition"] = d["sweep_value"].apply(lambda x: x["condition"])
        d["post_gap"] = d["sweep_value"].apply(lambda x: x["post_gap"])
        return d[(d["condition"] == "wrong") & (d["post_gap"] == 0)]
    rows += accuracy_headline_rows(df, "sycophancy", syc_cell)
    rows += sycophancy_did_rows(df)

    if not rows:
        print("no paired data available across experiments")
        return

    out = pd.DataFrame(rows)
    print("=== cross-experiment paired headline ===")
    print(out.to_string(index=False))
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv_out, index=False)
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--csv-out", default="data/headline.csv")
    main(p.parse_args())
