"""Plot accuracy vs. number of filler turns, faceted by filler kind."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_pathologies.analysis.metrics import load_jsonl
from agent_pathologies.analysis.plots import plot_accuracy_curve


def main(args: argparse.Namespace) -> None:
    df = load_jsonl(Path(args.path))
    df["n_filler"] = df["sweep_value"].apply(lambda x: x["n_filler"])
    df["kind"] = df["sweep_value"].apply(lambda x: x["kind"])

    grouped = (
        df.groupby(["kind", "n_filler"])["is_correct"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )
    print(grouped.to_string(index=False))

    out_dir = Path("data/plots")
    for kind in df["kind"].unique():
        sub = grouped[grouped["kind"] == kind].copy()
        plot_accuracy_curve(
            sub,
            x_col="n_filler",
            out=out_dir / f"context_rot_{kind}.png",
            title=f"Context rot — filler kind: {kind}",
        )
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/context_rot.jsonl")
    main(p.parse_args())
