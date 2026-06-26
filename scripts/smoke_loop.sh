#!/bin/bash
# 迷你 self-play loop 冒烟测试：2 轮，每轮 50 条数据 + 10 步训练 + HumanEval 评估
# 验证完整 生成→训练→评估→下一轮 链路
# 用法：sbatch scripts/smoke_loop.sh

#SBATCH --job-name=smoke_loop
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=02:00:00
#SBATCH --output=selfplay_results/logs/smoke_loop_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled

BASE_MODEL="bigcode/santacoder"
TOTAL_ROUNDS=2
STEPS_PER_ROUND=10
NUM_SAMPLES=50
ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"

SMOKE_DIR="selfplay_results/smoke_loop"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR/generated_data"

for ROUND in $(seq 1 $TOTAL_ROUNDS); do
    if [ $ROUND -eq 1 ]; then
        CURRENT_MODEL="$BASE_MODEL"
    else
        CURRENT_MODEL="$SMOKE_DIR/round$((ROUND-1))/final_checkpoint"
    fi

    echo ""
    echo "========== Round $ROUND / $TOTAL_ROUNDS =========="
    echo "模型: $CURRENT_MODEL"

    # ---- Step 1: 生成数据 ----
    DATA_FILE="$SMOKE_DIR/generated_data/round${ROUND}.jsonl"
    echo "--- Step 1: 生成 $NUM_SAMPLES 条数据 ---"
    python scripts/generate_data.py \
        --model_path "$CURRENT_MODEL" \
        --output_file "$DATA_FILE" \
        --num_samples "$NUM_SAMPLES" \
        --seed "$ROUND" \
        --local_dataset_path "$ARROW_CACHE" \
        --batch_size 16 \
        --prompt_tokens 1024 \
        --max_new_tokens 1024 \
        --temperature 0.8 \
        --top_p 0.95

    echo "生成了 $(wc -l < "$DATA_FILE") 条样本"

    # ---- Step 2: 训练 ----
    echo "--- Step 2: 训练 $STEPS_PER_ROUND 步 ---"
    torchrun --nproc_per_node 2 --standalone train.py \
        --local_data_path "$DATA_FILE" \
        --data_column content \
        --model_path "$CURRENT_MODEL" \
        --seq_length 2048 \
        --max_steps "$STEPS_PER_ROUND" \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --warmup_steps 2 \
        --weight_decay 0.05 \
        --fim_rate 0.5 \
        --fim_spm_rate 0.5 \
        --bf16 \
        --output_dir "$SMOKE_DIR/round${ROUND}" \
        --eval_freq 99999 \
        --save_freq 10 \
        --log_freq 5

    CKPT="$SMOKE_DIR/round${ROUND}/final_checkpoint"
    if [ ! -d "$CKPT" ]; then
        echo "FAIL: checkpoint $CKPT 不存在"
        exit 1
    fi

    # ---- Step 3: HumanEval 评估 ----
    echo "--- Step 3: HumanEval 评估 ---"
    SAMPLES="$SMOKE_DIR/round${ROUND}/humaneval_samples.jsonl"
    python scripts/evalplus_generate.py \
        --model_path "$CKPT" \
        --dataset humaneval \
        --output_file "$SAMPLES" \
        --temperature 0.2

    rm -f "${SAMPLES%.jsonl}_eval_results.json"
    evalplus.evaluate \
        --dataset humaneval \
        --samples "$SAMPLES" \
        --i-just-wanna-run

    echo "========== Round $ROUND 完成 =========="
done

echo ""
echo "=========================================="
echo "  迷你 loop 冒烟测试全部通过！"
echo "=========================================="
