#!/bin/bash
#SBATCH --job-name=v2_smoke_all
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=12:00:00
#SBATCH --output=v2_smoke_all_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

FILTER="${1:-none}"
OUT_ROOT="${2:-results_smoke}"

for MODEL in santacoder starcoder2 qwen25 codellama; do
    echo ""
    echo "============================================"
    echo "=== Smoke: $MODEL / $FILTER ==="
    echo "============================================"
    bash src/run_experiment.sh "$MODEL" "$FILTER" \
        --rounds 1 \
        --num-samples 8 \
        --raw-num-samples 32 \
        --max-steps 5 \
        --output-root "$OUT_ROOT" \
        --prompt-tokens 128 \
        --gen-max-new-tokens 128 \
        --gen-batch-size 2 \
        --train-seq-length 512 \
        --train-batch-size 1 \
        --grad-accum 1 \
        --warmup-steps 0 \
        --lcb-debug \
        --lcb-max-new-tokens 128
done

echo ""
echo "=== All model smoke tests done ==="
