"""Accuracy after wrong-pushback, as a function of turns between pushback and re-probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_pathologies.analysis.metrics import load_jsonl
from agent_pathologies.analysis.plots import plot_accuracy_curve


def main(args: argparse.Namespace) -> None:
    df = load_jsonl(Path(args.path))
    df["post_gap"] = df["sweep_value"].apply(lambda x: x["post_gap"])

    grouped = (
        df.groupby("post_gap")["is_correct"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    print("Accuracy at re-probe, by # neutral turns between pushback and re-probe:")
    print(grouped.to_string(index=False))

    out = Path("data/plots/sycophancy.png")
    plot_accuracy_curve(
        grouped,
        x_col="post_gap",
        out=out,
        title="Sycophancy persistence — accuracy at re-probe vs. post-pushback gap",
    )
    print(f"plot written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/sycophancy.jsonl")
    main(p.parse_args())
