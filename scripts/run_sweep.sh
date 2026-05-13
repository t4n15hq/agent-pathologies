#!/usr/bin/env bash
# Run all three experiments, config-driven, resumable.
#
#   bash scripts/run_sweep.sh                # full sweep (pairs + anchors)
#   bash scripts/run_sweep.sh --mock         # smoke test, no API
#   bash scripts/run_sweep.sh --anchors skip # stage 1: Chinese pairs only
#   bash scripts/run_sweep.sh --anchors only # stage 2: anchors only (after stage 1)
#
# Re-running this script after a crash skips already-completed cells.

set -euo pipefail

PY=".venv/bin/python"
MOCK_FLAG=""
ANCHORS_FLAG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)
            MOCK_FLAG="--mock"
            shift
            ;;
        --anchors)
            ANCHORS_FLAG+=("--anchors" "$2")
            shift 2
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

run_one () {
    local exp="$1"
    echo "=== $exp ==="
    $PY "experiments/${exp}/run.py" $MOCK_FLAG "${ANCHORS_FLAG[@]}" --concurrency 8
    $PY "experiments/${exp}/analyze.py"
    echo
}

run_one self_consistency
run_one context_rot
run_one sycophancy

echo "All sweeps done."
echo "Cross-experiment summary:"
$PY scripts/pair_analysis.py
