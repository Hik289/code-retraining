#!/bin/bash
# 训练流程冒烟测试：1 卡，10 步，验证能跑通
# 用法：sbatch scripts/test_train.sh

#SBATCH --job-name=test_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --output=selfplay_results/logs/test_train_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled

mkdir -p selfplay_results/logs test_train_output

ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"

echo "===== 训练冒烟测试 ====="

python train.py \
    --local_arrow_path "$ARROW_CACHE" \
    --data_column content \
    --seq_length 2048 \
    --max_steps 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 2 \
    --weight_decay 0.05 \
    --fim_rate 0.5 \
    --fim_spm_rate 0.5 \
    --bf16 \
    --output_dir test_train_output \
    --eval_freq 5 \
    --save_freq 10 \
    --log_freq 1

echo "===== 测试完成 ====="
