#!/bin/bash
# Submit V2 experiments in controlled batches.
#
# Usage:
#   bash src/submit_experiments.sh none
#   bash src/submit_experiments.sh compile
#   bash src/submit_experiments.sh quality
#   bash src/submit_experiments.sh ppl
#   bash src/submit_experiments.sh binary
#   bash src/submit_experiments.sh all

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "Usage: bash src/submit_experiments.sh {none|compile|quality|ppl|binary|all}"
    exit 1
fi

MODELS=(santacoder starcoder2 qwen25 codellama)
FILTERS=(none compile quality ppl binary)

submit_one() {
    local model="$1"
    local filter="$2"
    echo "Submitting: $model / $filter"
    sbatch --job-name="v2_${model}_${filter}" src/run_experiment.sh "$model" "$filter"
}

if [ "$TARGET" = "all" ]; then
    for filter in "${FILTERS[@]}"; do
        for model in "${MODELS[@]}"; do
            submit_one "$model" "$filter"
        done
    done
else
    case "$TARGET" in
        none|compile|quality|ppl|binary) ;;
        *)
            echo "ERROR: unknown target '$TARGET'"
            echo "Usage: bash src/submit_experiments.sh {none|compile|quality|ppl|binary|all}"
            exit 1
            ;;
    esac

    for model in "${MODELS[@]}"; do
        submit_one "$model" "$TARGET"
    done
fi

echo "Done. Current queue:"
squeue -u $USER
