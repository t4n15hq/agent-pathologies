from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .stats import bootstrap_ci


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [load_jsonl(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_analyzable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded rows. Use the analyzable subset for hypothesis testing
    (excluded counts are reported separately per preregistration §6)."""
    if df.empty:
        return df
    if "excluded" in df.columns:
        excluded = df["excluded"].fillna(False).astype(bool)
    else:
        excluded = pd.Series(False, index=df.index, dtype=bool)
    mask = (~excluded) & df["is_correct"].notna()
    return df.loc[mask].copy()


def accuracy_with_ci(values: Iterable[bool], n_iters: int = 10_000) -> dict:
    bool_vals = [bool(v) for v in values]
    if not bool_vals:
        return {"accuracy": float("nan"), "n": 0, "ci_lo": float("nan"), "ci_hi": float("nan")}
    acc = sum(bool_vals) / len(bool_vals)
    floats = [1.0 if v else 0.0 for v in bool_vals]
    lo, hi = bootstrap_ci(floats, n_iters=n_iters)
    return {"accuracy": acc, "n": len(bool_vals), "ci_lo": lo, "ci_hi": hi}


def accuracy_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = accuracy_with_ci(grp["is_correct"].tolist())
        rows.append({**dict(zip(group_cols, keys)), **stats})
    return pd.DataFrame(rows)


def answer_divergence(answers: Iterable[str]) -> float:
    """Fraction of runs that disagree with the modal answer.
    0.0 = perfect consistency; approaches 1.0 as no answer dominates."""
    cleaned = [(a or "").strip().lower() for a in answers if a]
    if not cleaned:
        return 0.0
    counts: dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    mode_freq = max(counts.values())
    return 1.0 - (mode_freq / len(cleaned))


def exploratory_families(df: pd.DataFrame) -> set[str]:
    """Return the set of model_family values where any trajectory carries
    `extra.exploratory == True`. Used by analyzers to tag rows."""
    if df.empty or "extra" not in df.columns:
        return set()
    mask = df["extra"].apply(lambda x: bool(isinstance(x, dict) and x.get("exploratory")))
    if not mask.any():
        return set()
    return set(df.loc[mask, "model_family"].dropna().unique())


def tag_exploratory(family: str, exploratory: set[str]) -> str:
    """Append a [exploratory] tag if the family is in the exploratory set.
    Idempotent and safe to call on already-tagged strings."""
    if family in exploratory and not family.endswith("[exploratory]"):
        return f"{family} [exploratory]"
    return family


def exclusion_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report exclusion counts per (model, exclusion_reason)."""
    if df.empty or "excluded" not in df.columns:
        return pd.DataFrame()
    ex = df[df["excluded"] == True]
    if ex.empty:
        return pd.DataFrame()
    return (
        ex.groupby(["model", "exclusion_reason"])
        .size()
        .reset_index(name="n_excluded")
    )
