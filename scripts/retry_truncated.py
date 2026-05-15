"""Drop `truncated_at_max_tokens` rows from JSONL files so they get
re-attempted on the next sweep run. Pair with a higher `--max-tokens` on
the experiment run.py to actually recover the cell.

USAGE:
    # 1. Drop the truncated rows from the existing data files
    python scripts/retry_truncated.py --apply

    # 2. Re-run each experiment with higher max_tokens — resumability
    #    will only attempt the dropped cells, not the ones already done.
    python experiments/self_consistency/run.py --max-tokens 4096
    python experiments/sycophancy/run.py --max-tokens 4096
    python experiments/context_rot/run.py --max-tokens 4096

Default behavior is DRY-RUN: only reports counts, doesn't modify files.
Pass --apply to actually rewrite. A backup of each modified file is
written to data/<exp>.jsonl.pre-retry-truncated for safety.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def process(path: Path, apply: bool) -> dict:
    rows = []
    n_truncated = 0
    n_total = 0
    truncated_by_model = Counter()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            if (r.get("exclusion_reason") or "").startswith("truncated_at_max_tokens"):
                n_truncated += 1
                truncated_by_model[(r["model"], r.get("model_role"))] += 1
                continue  # drop this row
            rows.append(r)

    print(f"  {path.name}: {n_total} total, {n_truncated} truncated → dropping")
    for (model, role), n in truncated_by_model.most_common():
        print(f"    {model[:55]:<55s} role={role:<10s} n={n}")

    if not apply:
        return {"path": str(path), "total": n_total, "truncated": n_truncated}

    backup = path.with_suffix(path.suffix + ".pre-retry-truncated")
    shutil.copyfile(path, backup)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"    wrote {len(rows)} rows back to {path.name}; backup at {backup.name}")
    return {"path": str(path), "total": n_total, "truncated": n_truncated}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--apply", action="store_true",
                   help="Actually rewrite the JSONL files (default is dry-run).")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"=== retry_truncated.py ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"scanning {data_dir}/*.jsonl")
    print()

    summary = []
    for path in sorted(data_dir.glob("*.jsonl")):
        if "_clean" in path.name or "pre-retry" in path.name:
            continue
        summary.append(process(path, args.apply))
        print()

    total_truncated = sum(s["truncated"] for s in summary)
    print(f"--- TOTAL: {total_truncated} truncated rows {'dropped' if args.apply else 'would be dropped'} ---")
    if not args.apply:
        print("Re-run with --apply to actually modify files.")


if __name__ == "__main__":
    main()
