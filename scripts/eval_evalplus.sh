#!/bin/bash
# EvalPlus 评估（HumanEval+ 和 MBPP+）
# 流程：自定义 codegen 生成代码 → evalplus.evaluate 评估
# 用法：sbatch scripts/eval_evalplus.sh <model_path> [humaneval|mbpp]
# 默认同时评估 humaneval 和 mbpp

#SBATCH --job-name=eval_evalplus
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=02:00:00
#SBATCH --output=selfplay_results/logs/evalplus_%j.out

set -euo pipefail

# ---- 项目根目录 ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

# ---- 环境 ----
source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled

MODEL_PATH="${1:-bigcode/santacoder}"
DATASETS="${2:-humaneval mbpp}"
RESULTS_DIR="./evalplus_results"

mkdir -p "$RESULTS_DIR" selfplay_results/logs

MODEL_ID=$(echo "$MODEL_PATH" | sed 's|^\./||' | sed 's|/|--|g')

for DATASET in $DATASETS; do
    SAMPLES="${RESULTS_DIR}/${DATASET}/${MODEL_ID}_greedy.jsonl"
    mkdir -p "${RESULTS_DIR}/${DATASET}"

    echo "===== 生成 $DATASET 代码 ====="
    python scripts/evalplus_generate.py \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --output_file "$SAMPLES" \
        --temperature 0.2

    echo "===== 评估 $DATASET ====="
    # 删除旧结果文件，避免 evalplus 交互式 overwrite 提示导致 EOFError
    EVAL_RESULT="${SAMPLES%.jsonl}_eval_results.json"
    rm -f "$EVAL_RESULT"
    evalplus.evaluate \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --i-just-wanna-run
    echo "===== $DATASET 完成 ====="
done