#!/usr/bin/env bash
# Run all three experiments, config-driven.
#
#   bash scripts/run_sweep.sh           # full sweep against configs/models.yaml
#   bash scripts/run_sweep.sh --mock    # smoke test, no API
#
# Resumable: re-running this script after a crash skips already-completed cells.

set -euo pipefail

PY=".venv/bin/python"
MOCK_FLAG=""
if [[ "${1-}" == "--mock" ]]; then
    MOCK_FLAG="--mock"
fi

run_one () {
    local exp="$1"
    echo "=== $exp ==="
    $PY "experiments/${exp}/run.py" $MOCK_FLAG --concurrency 8
    $PY "experiments/${exp}/analyze.py"
    echo
}

# Self-consistency first — gives the noise floor for the other two.
run_one self_consistency
run_one context_rot
run_one sycophancy

echo "All sweeps done."
echo "Cross-experiment summary:"
$PY scripts/pair_analysis.py
