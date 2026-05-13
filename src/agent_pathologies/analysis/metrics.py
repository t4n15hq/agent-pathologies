from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def accuracy_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)["is_correct"]
        .agg(["mean", "count", "sem"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .reset_index()
    )


def answer_divergence(answers: Iterable[str]) -> float:
    """Fraction of runs that disagree with the modal answer.
    0.0 = every run gave the same answer (perfectly consistent).
    Approaches 1.0 as no answer dominates."""
    cleaned = [(a or "").strip().lower() for a in answers]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return 0.0
    counts: dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    mode_freq = max(counts.values())
    return 1.0 - (mode_freq / len(cleaned))
