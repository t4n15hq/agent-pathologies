"""Compute per-task answer divergence and accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_pathologies.analysis.metrics import answer_divergence, load_jsonl


def main(args: argparse.Namespace) -> None:
    df = load_jsonl(Path(args.path))
    per_task = (
        df.groupby("task_id")
        .agg(
            divergence=("probe_answer", answer_divergence),
            accuracy=("is_correct", "mean"),
            n=("trajectory_id", "count"),
        )
        .reset_index()
    )

    print("per-task results:")
    print(per_task.to_string(index=False))
    print()
    print(f"mean divergence across tasks: {per_task['divergence'].mean():.3f}")
    print(f"mean accuracy across tasks:   {per_task['accuracy'].mean():.3f}")
    print(f"max  divergence:              {per_task['divergence'].max():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/self_consistency.jsonl")
    main(p.parse_args())
