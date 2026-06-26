#!/bin/bash
# LiveCodeBench 评估（自定义 codegen，绕过 vLLM）
# 用法：sbatch scripts/eval_livecodebench.sh <model_path> [release_version]
# 默认 release_v1

#SBATCH --job-name=eval_lcb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=04:00:00
#SBATCH --output=selfplay_results/logs/lcb_%j.out

set -euo pipefail

# ---- 项目根目录 ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

# ---- 环境 ----
source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WANDB_MODE=disabled

MODEL_PATH="${1:-bigcode/santacoder}"
VERSION="${2:-release_v1}"
RESULTS_DIR="./livecodebench_results"

mkdir -p "$RESULTS_DIR" selfplay_results/logs

echo "===== LiveCodeBench: codegeneration ($VERSION) ====="
python scripts/lcb_generate.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$RESULTS_DIR" \
    --release_version "$VERSION" \
    --temperature 0.2 \
    --max_new_tokens 512 \
    --n_samples 1 \
    --num_process_evaluate 16 \
    --timeout 6
echo "===== 完成 ====="
