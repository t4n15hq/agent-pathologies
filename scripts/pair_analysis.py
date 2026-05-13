"""One-shot cross-experiment summary. Loads all three JSONL outputs,
computes the headline paired numbers per family, prints + writes CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_pathologies.analysis.metrics import filter_analyzable, load_jsonl
from agent_pathologies.analysis.stats import paired_test


def headline_row(df: pd.DataFrame, experiment: str, cell_filter):
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
            "n_paired": res.n,
            "acc_instruct": res.p_a,
            "acc_reasoning": res.p_b,
            "cohens_h": res.cohens_h,
            "p_value": res.p_value,
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
    # self_consistency: cell = all-replays-for-task → not directly comparable
    # in is_correct form; we just compare marginal accuracy here.
    rows += headline_row(df, "self_consistency", lambda d: d)

    # context_rot: headline cell is largest filler count, irrelevant
    def ctx_cell(d):
        d = d.copy()
        d["n_filler"] = d["sweep_value"].apply(lambda x: x["n_filler"])
        d["kind"] = d["sweep_value"].apply(lambda x: x["kind"])
        max_k = d["n_filler"].max()
        return d[(d["kind"] == "irrelevant") & (d["n_filler"] == max_k)]
    rows += headline_row(df, "context_rot", ctx_cell)

    # sycophancy: headline cell is condition=wrong, gap=0
    def syc_cell(d):
        d = d.copy()
        d["condition"] = d["sweep_value"].apply(lambda x: x["condition"])
        d["post_gap"] = d["sweep_value"].apply(lambda x: x["post_gap"])
        return d[(d["condition"] == "wrong") & (d["post_gap"] == 0)]
    rows += headline_row(df, "sycophancy", syc_cell)

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
