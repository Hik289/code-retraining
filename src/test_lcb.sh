#!/bin/bash
#SBATCH --job-name=test_lcb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --output=test_lcb_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

OUT_DIR="/tmp/test_lcb"
mkdir -p "$OUT_DIR"

echo "=== SantaCoder LiveCodeBench debug test ==="
python src/evaluate_lcb.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --output_dir "$OUT_DIR" \
    --release_version release_v1 \
    --temperature 0.0 \
    --n_samples 1 \
    --max_new_tokens 256 \
    --num_process_evaluate 16 \
    --timeout 6 \
    --debug

echo ""
echo "=== Scores ==="
cat "$OUT_DIR"/santacoder_bigcode--santacoder_release_v1_debug_scores.json
echo ""
echo "=== LiveCodeBench debug test done ==="
