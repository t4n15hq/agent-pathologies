#!/bin/bash
# Live progress monitor for agent-pathologies sweeps and retries.
# Refreshes every 20s. Ctrl-C to exit.
#
# Usage:
#   bash scripts/watch_progress.sh
#   bash scripts/watch_progress.sh 10   # refresh every 10s instead

REFRESH="${1:-20}"
cd "$(dirname "$0")/.."

while :; do
  clear
  printf '\033[1m=== agent-pathologies live progress  (%s, refresh %ss) ===\033[0m\n\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$REFRESH"

  # Process status
  printf '\033[1mProcesses:\033[0m\n'
  for label in \
      "retry_deepseek_empties|DeepSeek empties retry (DS-direct)" \
      "retry_v4pro_instruct.*--axis self_consistency|V4-pro instruct retry [SC] (DS-direct)" \
      "retry_v4pro_instruct.*--axis context_rot|V4-pro instruct retry [CT] (DS-direct)" \
      "retry_v4pro_reasoning_provider_error|V4-pro reasoning prov_err retry (DS-direct)" \
      "retry_qwen_truncated|Qwen truncated retry (OR)" \
      "context_rot/run.py|context_rot sweep/retry (OR)" \
      "sycophancy/run.py|sycophancy sweep/retry (OR)" \
      "self_consistency/run.py|self_consistency sweep/retry (OR)"; do
    pattern="${label%%|*}"; name="${label##*|}"
    pid=$(pgrep -f "$pattern" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
      etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
      printf "  \033[32m●\033[0m %-40s  PID %s  uptime %s\n" "$name" "$pid" "$etime"
    else
      printf "  \033[90m○\033[0m %-40s  idle\n" "$name"
    fi
  done

  printf '\n\033[1mAxis state (post-dedup, effective counts):\033[0m\n'
  .venv/bin/python <<'PY' 2>/dev/null
import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from agent_pathologies.analysis.metrics import load_jsonl, filter_analyzable, exclusion_report

PLAN = {"self_consistency": 8000, "sycophancy": 4800, "context_rot": 9600}
print(f"  {'axis':<18}  {'analyzable':>16}  {'%':>6}  {'effective excl':>14}  {'raw rows':>9}")
print(f"  {'-'*18}  {'-'*16}  {'-'*6}  {'-'*14}  {'-'*9}")
for axis, plan in PLAN.items():
    p = Path(f"data/{axis}.jsonl")
    if not p.exists():
        continue
    df = load_jsonl(p)
    af = filter_analyzable(df)
    er = exclusion_report(df)
    n_excl = int(er["n_excluded"].sum()) if not er.empty else 0
    pct = 100 * len(af) / plan
    bar_width = 20
    filled = int(bar_width * len(af) / plan)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {axis:<18}  {len(af):>6} / {plan:<5}   {pct:5.1f}%  {n_excl:>14}  {len(df):>9}")
    print(f"  {'':<18}  {bar}")
PY

  printf '\n\033[1mLast JSONL writes:\033[0m\n'
  for f in data/self_consistency.jsonl data/sycophancy.jsonl data/context_rot.jsonl; do
    if [ -f "$f" ]; then
      mtime=$(stat -f '%Sm' -t '%H:%M:%S' "$f")
      printf '  %-30s  %s\n' "$f" "$mtime"
    fi
  done

  printf '\n\033[2m  (Ctrl-C to exit; refresh every %ss; pass an arg to change: bash scripts/watch_progress.sh 10)\033[0m\n' "$REFRESH"
  sleep "$REFRESH"
done
