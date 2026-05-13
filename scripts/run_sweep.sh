#!/usr/bin/env bash
# Run all three experiments across the recommended 4-model sweep.
# Output JSONL files are namespaced by model so analysis can join across them.
#
# Usage:
#   bash scripts/run_sweep.sh                # full v1 sweep
#   bash scripts/run_sweep.sh --quick        # smaller (5 tasks) for sanity
#
# Requires OPENROUTER_API_KEY in env (or .env loaded).

set -euo pipefail

PY=".venv/bin/python"
PROVIDER="openrouter"
N_TASKS=30
N_REPEATS=20

if [[ "${1-}" == "--quick" ]]; then
    N_TASKS=5
    N_REPEATS=5
fi

MODELS=(
    "qwen/qwen3.5-397b-a17b"
    "deepseek/deepseek-v4-pro"
    "z-ai/glm-4.7"
    "anthropic/claude-opus-4.7"
)

slug() {
    echo "$1" | tr '/.' '__'
}

for m in "${MODELS[@]}"; do
    s=$(slug "$m")
    echo "=== $m ==="

    echo "--- self_consistency ---"
    $PY experiments/self_consistency/run.py \
        --provider "$PROVIDER" --model "$m" \
        --n-tasks "$N_TASKS" --n-repeats "$N_REPEATS" \
        --out "data/${s}__self_consistency.jsonl"

    echo "--- context_rot ---"
    $PY experiments/context_rot/run.py \
        --provider "$PROVIDER" --model "$m" \
        --n-tasks "$N_TASKS" \
        --out "data/${s}__context_rot.jsonl"

    echo "--- sycophancy ---"
    $PY experiments/sycophancy/run.py \
        --provider "$PROVIDER" --model "$m" \
        --n-tasks "$N_TASKS" \
        --out "data/${s}__sycophancy.jsonl"
done

echo
echo "All sweeps complete. Trajectories in data/"
