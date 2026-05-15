"""Produce a clean version of each JSONL with only analyzable rows
(excluded == False AND is_correct is not None), retroactively reclassifying
legacy `empty_probe_answer` → `provider_empty_response` per the metrics
helper. Writes to data/<exp>_clean.jsonl.

USAGE:
    python scripts/clean_dataset.py
    # then point analyzers / plots / paper figures at the _clean files

This is purely a derived artifact — no original data is modified.
Useful for paper figures and shareable datasets where you don't want
reviewers to have to think about exclusion classes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_pathologies.analysis.metrics import (
    dedupe_to_latest,
    reclassify_legacy_exclusions,
)
import pandas as pd


def clean_one(in_path: Path, out_path: Path) -> dict:
    rows = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"  {in_path.name}: empty, skipping")
        return {"in": str(in_path), "out": str(out_path), "n_in": 0, "n_out": 0}

    df = pd.DataFrame(rows)
    df = reclassify_legacy_exclusions(df)
    # Semantic dedup BEFORE filtering: collapses retry rows for the same
    # logical cell to a single representative row (prefer non-excluded;
    # tie-break to latest). Without this, a cell with [excluded, ok, ok]
    # produces two output rows. Matches the exact pipeline used by
    # `filter_analyzable` so paper tables and the released clean dataset
    # are bit-for-bit consistent.
    df = dedupe_to_latest(df)

    excluded = df["excluded"].fillna(False).astype(bool) if "excluded" in df.columns \
               else pd.Series(False, index=df.index, dtype=bool)
    has_score = df["is_correct"].notna() if "is_correct" in df.columns \
               else pd.Series(False, index=df.index, dtype=bool)
    keep_mask = (~excluded) & has_score

    kept_rows = df.loc[keep_mask].to_dict(orient="records")
    exclusion_counts = Counter(
        (r.get("exclusion_reason") or "kept") for r in df.to_dict(orient="records")
        if r.get("excluded") or not r.get("is_correct") is not None or not r.get("excluded")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")

    print(f"  {in_path.name} → {out_path.name}: {len(rows)} → {len(kept_rows)} analyzable "
          f"({100*len(kept_rows)/len(rows):.0f}% kept)")

    # Print exclusion breakdown for the dropped rows
    dropped = df.loc[~keep_mask]
    if not dropped.empty:
        dropped_reasons = Counter(dropped.get("exclusion_reason", pd.Series()).fillna("missing_is_correct").tolist())
        for reason, n in dropped_reasons.most_common(5):
            print(f"    dropped {n:>5d}: {reason[:60]}")

    return {"in": str(in_path), "out": str(out_path),
            "n_in": len(rows), "n_out": len(kept_rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"=== clean_dataset.py ===")
    print(f"scanning {data_dir}/*.jsonl (excluding _clean and pre-retry files)")
    print()

    summary = []
    for path in sorted(data_dir.glob("*.jsonl")):
        if "_clean" in path.name or "pre-retry" in path.name:
            continue
        out = data_dir / path.name.replace(".jsonl", "_clean.jsonl")
        summary.append(clean_one(path, out))
        print()

    total_in = sum(s["n_in"] for s in summary)
    total_out = sum(s["n_out"] for s in summary)
    print(f"--- TOTAL: {total_in} → {total_out} analyzable "
          f"({100*total_out/total_in:.0f}% kept) ---")


if __name__ == "__main__":
    main()
