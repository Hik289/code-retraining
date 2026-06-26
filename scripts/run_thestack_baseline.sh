#!/bin/bash
# The Stack baseline（30K 步，Python，FIM 0.5，2 卡数据并行）— self-play 的参照上限
# 用法：sbatch scripts/run_thestack_baseline.sh

#SBATCH --job-name=thestack_baseline
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=24:00:00
#SBATCH --output=selfplay_results/logs/baseline_%j.out

set -euo pipefail

# ---- 项目根目录 ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

# ---- 环境 ----
source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled

mkdir -p selfplay_results/logs thestack_baseline

ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"

echo "===== The Stack Baseline 训练 ====="
echo "GPU 数: 2"
echo "等效全局 batch: 2 GPU × 2 batch × 4 grad_accum = 16"
echo "总步数: 30000"

# 2 卡数据并行：grad_accum=4（等效全局 batch = 2×2×4 = 16）
torchrun --nproc_per_node 2 --standalone train.py \
    --local_arrow_path "$ARROW_CACHE" \
    --data_column content \
    --seq_length 2048 \
    --max_steps 30000 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --weight_decay 0.05 \
    --fim_rate 0.5 \
    --fim_spm_rate 0.5 \
    --bf16 \
    --output_dir thestack_baseline \
    --eval_freq 3000 \
    --save_freq 3000 \
    --log_freq 100

echo "===== Baseline 训练完成 ====="
