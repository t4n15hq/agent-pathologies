from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .stats import bootstrap_ci
from ..tasks.scoring import extract_last_integer


def _analysis_cell_key(model_family, task_id, sweep_value, seed, model_role) -> str:
    """Cell key used for ANALYSIS-time dedup.

    Intentionally broader than `runner.cell_key`: uses `model_family` rather
    than the raw `model` string. This lets the dedup pipeline collapse rows
    where the same logical (family, role, task, sweep, seed) cell was served
    by different providers — e.g. V4-pro initially served by OpenRouter/Novita
    (`model = "deepseek/deepseek-v4-pro"`) and re-served by DeepSeek-direct
    in the 2026-05-15 retry (`model = "deepseek-v4-pro"`). At the model layer
    these are the same model + same configuration; only the API gateway
    differs. Collapsing them at analysis time prevents double-counting OR
    rows as excluded while the DS-direct retry of the same cell shows as
    analyzable. Runner's cell_key remains stricter (it includes the raw
    model string) to preserve audit-trail granularity on the JSONL."""
    role = getattr(model_role, "value", model_role)
    payload = json.dumps({
        "model_family": str(model_family) if model_family is not None and not (isinstance(model_family, float) and pd.isna(model_family)) else None,
        "model_role": str(role) if role is not None and not (isinstance(role, float) and pd.isna(role)) else None,
        "task": task_id,
        "sweep": sweep_value,
        "seed": seed,
    }, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# Back-compat alias: external code/tests may still import _cell_key.
_cell_key = _analysis_cell_key


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


def reclassify_legacy_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Retroactively split legacy `empty_probe_answer` rows into the more
    specific `provider_empty_response` class where the underlying signal is
    output_tokens == 0 + empty content, AND canonicalize the noisy
    `provider_error:*` strings (which include Python object reprs with
    pointer addresses that vary every run) into the single class
    `provider_error`. Idempotent. Returns a copy."""
    if df.empty or "exclusion_reason" not in df.columns:
        return df
    out = df.copy()
    pa = out.get("probe_answer", pd.Series([None] * len(out)))
    out_tok = pd.to_numeric(out.get("output_tokens"), errors="coerce").fillna(-1)
    reason = out["exclusion_reason"].fillna("")
    is_legacy_empty = reason == "empty_probe_answer"
    truly_empty = is_legacy_empty & (out_tok == 0) & (pa.isna() | (pa.astype(str).str.strip() == ""))
    out.loc[truly_empty, "exclusion_reason"] = "provider_empty_response"
    # Canonicalize `provider_error:RetryError[<Future at 0x...>]` etc. to plain
    # `provider_error` so the §6 exclusion table aggregates correctly.
    is_provider_error = reason.str.startswith("provider_error")
    out.loc[is_provider_error, "exclusion_reason"] = "provider_error"
    return out


def dedupe_to_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows for the same trajectory cell, preferring
    non-excluded over excluded.

    The JSONL is append-only and resumability re-attempts cells classified
    as retry-eligible (`provider_error:*`) or recovered by amendment (e.g.
    the 2026-05-15 max_tokens-bug-fix retry of DeepSeek-reasoning empties).
    Both the original failed row and the subsequent successful row remain
    in the file. For analysis we keep one row per cell, preferring the
    successful retry if any exists.

    This does NOT alter the JSONL — old rows stay on disk as an audit trail.
    Callers that need an exclusion *count* of cells where every attempt
    failed should pipe through this first; `exclusion_report` does so."""
    if df.empty:
        return df
    out = df.copy()
    # Use the analysis-layer cell key (keyed on model_family, not the raw
    # model string) so that OR-served and DS-direct-served rows of the same
    # logical cell collapse together. See `_analysis_cell_key` docstring.
    out["_cell_key"] = out.apply(
        lambda r: _analysis_cell_key(
            r.get("model_family"), r.get("task_id"), r.get("sweep_value"),
            r.get("seed"), r.get("model_role")),
        axis=1,
    )
    excluded = out.get("excluded", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["_excluded_sort"] = excluded.astype(int)  # False (0) sorts before True (1)
    # Preserve append order for tiebreaks: among rows with same excluded status,
    # keep the latest (most recent retry attempt).
    out["_orig_idx"] = range(len(out))
    out = out.sort_values(["_cell_key", "_excluded_sort", "_orig_idx"],
                           ascending=[True, True, False])
    out = out.drop_duplicates(subset=["_cell_key"], keep="first")
    return out.drop(columns=["_cell_key", "_excluded_sort", "_orig_idx"]).copy()


def filter_analyzable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded rows. Use the analyzable subset for hypothesis testing
    (excluded counts are reported separately per preregistration §6).
    Retroactively canonicalizes `empty_probe_answer` → `provider_empty_response`
    where the underlying signal warrants the more specific class. Dedupes
    by cell_key so re-attempted cells (e.g. the max_tokens-bug retry) don't
    double-count: the successful row supersedes the failed one for analysis."""
    if df.empty:
        return df
    df = reclassify_legacy_exclusions(df)
    df = dedupe_to_latest(df)
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
    """Fraction of runs that disagree with the modal answer over the raw
    response strings. Sensitive to surface-form variation (e.g., chain-of-
    thought length). For arithmetic tasks where the scored quantity is an
    integer, prefer `extracted_divergence(answers, extract_last_integer)`."""
    cleaned = [(a or "").strip().lower() for a in answers if a]
    if not cleaned:
        return 0.0
    counts: dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    mode_freq = max(counts.values())
    return 1.0 - (mode_freq / len(cleaned))


def extracted_divergence(
    answers: Iterable[str],
    extractor: Callable[[str], object] = extract_last_integer,
) -> float:
    """Same as `answer_divergence` but compares the EXTRACTED scored quantity
    (default: the last integer in each response) rather than the full string.

    This is the published reporting metric for self_consistency on arithmetic
    tasks: it isolates "did the model commit to the same answer" from the
    confound of varying chain-of-thought text. Per the analysis-agent finding
    that v4-flash divergence on raw strings was inflated by CoT length
    variability."""
    extracted = []
    for a in answers:
        v = extractor(a or "")
        if v is None:
            continue
        extracted.append(str(v))
    if not extracted:
        return 0.0
    counts: dict[str, int] = {}
    for x in extracted:
        counts[x] = counts.get(x, 0) + 1
    mode_freq = max(counts.values())
    return 1.0 - (mode_freq / len(extracted))


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
    """Report exclusion counts per (model, exclusion_reason).

    Dedupes by cell_key first so superseded rows are not counted. A cell is
    only reported as excluded if EVERY attempt for that cell failed — cells
    where a retry recovered the trajectory show up as analyzable, not as
    excluded. This matches what reviewers want: the §6 table reports the
    final state of the dataset, not transient failures along the way."""
    if df.empty or "excluded" not in df.columns:
        return pd.DataFrame()
    df = reclassify_legacy_exclusions(df)
    df = dedupe_to_latest(df)
    ex = df[df["excluded"] == True]
    if ex.empty:
        return pd.DataFrame()
    return (
        ex.groupby(["model", "exclusion_reason"])
        .size()
        .reset_index(name="n_excluded")
    )
