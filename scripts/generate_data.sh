#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=04:00:00
#SBATCH --output=selfplay_results/logs/generate_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

MODEL_PATH="${1:-bigcode/santacoder}"
OUTPUT_FILE="${2:-selfplay_results/generated_data/round1.jsonl}"
NUM_SAMPLES="${3:-5000}"
SEED="${4:-1}"

mkdir -p "$(dirname "$OUTPUT_FILE")" selfplay_results/logs

echo "===== 数据生成 ====="
echo "模型: $MODEL_PATH"
echo "输出: $OUTPUT_FILE"
echo "样本数: $NUM_SAMPLES"
echo "Seed: $SEED"

ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"

python scripts/generate_data.py \
    --model_path "$MODEL_PATH" \
    --output_file "$OUTPUT_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --local_dataset_path "$ARROW_CACHE" \
    --batch_size 8 \
    --prompt_tokens 1024 \
    --max_new_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95

echo "===== 数据生成完成 ====="
echo "文件行数: $(wc -l < "$OUTPUT_FILE")"
