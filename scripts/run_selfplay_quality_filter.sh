#!/bin/bash
# Exp B: Self-Play + Compile + Quality Filter (重复度 + 长度)
# 每轮循环生成+两级过滤，直到达到 5000 条，然后训练+评估
# 用法：sbatch scripts/run_selfplay_quality_filter.sh

#SBATCH --job-name=sp_quality
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=72:00:00
#SBATCH --output=quality_filter_%j.out

set -euo pipefail

# ---- 项目根目录 ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

# ---- 环境 ----
source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ---- 配置 ----
BASE_MODEL="bigcode/santacoder"
TOTAL_ROUNDS=20
STEPS_PER_ROUND=3000
NUM_SAMPLES=5000
ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"
EXP_DIR="selfplay_results/quality_filter"

# ---- 初始化 ----
mkdir -p "$EXP_DIR/generated_data" selfplay_results/logs

CSV="$EXP_DIR/results.csv"
if [ ! -f "$CSV" ]; then
    echo "round,model_path,steps_total,humaneval_pass1,humaneval_plus_pass1,mbpp_pass1,mbpp_plus_pass1,livecodebench_pass1,filter_pass_rate,timestamp" > "$CSV"
fi

# ---- Self-Play 循环 ----
for ROUND in $(seq 1 $TOTAL_ROUNDS); do
    PREV_ROUND=$((ROUND - 1))

    if [ $ROUND -eq 1 ]; then
        CURRENT_MODEL="$BASE_MODEL"
        WARMUP=500
    else
        CURRENT_MODEL="$EXP_DIR/round${PREV_ROUND}/final_checkpoint"
        WARMUP=0
    fi

    STEPS_TOTAL=$((ROUND * STEPS_PER_ROUND))
    echo ""
    echo "========== [Exp B: Quality Filter] Round $ROUND / $TOTAL_ROUNDS (累计 $STEPS_TOTAL 步) =========="
    echo "模型: $CURRENT_MODEL"

    # ---- Step 1: 生成 + compile + quality 过滤（循环直到够数） ----
    DATA_FILE="$EXP_DIR/generated_data/round${ROUND}.jsonl"
    if [ -f "$DATA_FILE" ] && [ "$(wc -l < "$DATA_FILE")" -ge "$NUM_SAMPLES" ]; then
        echo "--- Step 1: 跳过（$DATA_FILE 已存在，$(wc -l < "$DATA_FILE") 条样本）---"
    else
        echo "--- Step 1: 生成 + compile+quality 过滤，目标 $NUM_SAMPLES 条 ---"
        python scripts/generate_data_filtered.py \
            --model_path "$CURRENT_MODEL" \
            --output_file "$DATA_FILE" \
            --num_samples "$NUM_SAMPLES" \
            --seed "$ROUND" \
            --local_dataset_path "$ARROW_CACHE" \
            --batch_size 128 \
            --prompt_tokens 1024 \
            --max_new_tokens 1024 \
            --temperature 0.8 \
            --top_p 0.95 \
            --filter_mode compile+quality \
            --repetition_threshold 0.5 \
            --min_completion_tokens 50
    fi
    LINES=$(wc -l < "$DATA_FILE")
    echo "数据: $LINES 条样本（过滤后）"

    # ---- Step 2: 训练 6000 步（2卡数据并行） ----
    CKPT="$EXP_DIR/round${ROUND}/final_checkpoint"
    if [ -d "$CKPT" ] && [ -f "$CKPT/model.safetensors" ]; then
        echo "--- Step 2: 跳过（$CKPT 已存在）---"
    else
        echo "--- Step 2: 训练 $STEPS_PER_ROUND 步 ---"
        torchrun --nproc_per_node 1 --standalone train.py \
            --local_data_path "$DATA_FILE" \
            --data_column content \
            --model_path "$CURRENT_MODEL" \
            --seq_length 2048 \
            --max_steps "$STEPS_PER_ROUND" \
            --batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate 5e-5 \
            --lr_scheduler_type cosine \
            --warmup_steps "$WARMUP" \
            --weight_decay 0.05 \
            --fim_rate 0.5 \
            --fim_spm_rate 0.5 \
            --bf16 \
            --output_dir "$EXP_DIR/round${ROUND}" \
            --eval_freq 1000 \
            --save_freq "$STEPS_PER_ROUND" \
            --log_freq 100
    fi

    if [ ! -d "$CKPT" ]; then
        echo "ERROR: checkpoint $CKPT 不存在，终止循环"
        exit 1
    fi

    # ---- Step 3: 评估 ----
    echo "--- Step 3: 评估 ---"

    # EvalPlus
    echo "--- 3a: EvalPlus ---"
    mkdir -p evalplus_results/humaneval evalplus_results/mbpp

    python scripts/evalplus_generate.py \
        --model_path "$CKPT" \
        --dataset humaneval \
        --output_file "evalplus_results/humaneval/quality_r${ROUND}_greedy.jsonl" \
        --temperature 0.2

    rm -f "evalplus_results/humaneval/quality_r${ROUND}_greedy_eval_results.json"
    EVAL_OUT=$(evalplus.evaluate \
        --dataset humaneval \
        --samples "evalplus_results/humaneval/quality_r${ROUND}_greedy.jsonl" \
        --i-just-wanna-run 2>&1 | tee /dev/stderr)
    HE_BASE=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | head -1 | awk '{print $NF}')
    HE_PLUS=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | tail -1 | awk '{print $NF}')
    echo "HumanEval pass@1: $HE_BASE | HumanEval+ pass@1: $HE_PLUS"

    python scripts/evalplus_generate.py \
        --model_path "$CKPT" \
        --dataset mbpp \
        --output_file "evalplus_results/mbpp/quality_r${ROUND}_greedy.jsonl" \
        --temperature 0.2

    rm -f "evalplus_results/mbpp/quality_r${ROUND}_greedy_eval_results.json"
    EVAL_OUT=$(evalplus.evaluate \
        --dataset mbpp \
        --samples "evalplus_results/mbpp/quality_r${ROUND}_greedy.jsonl" \
        --i-just-wanna-run 2>&1 | tee /dev/stderr)
    MBPP_BASE=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | head -1 | awk '{print $NF}')
    MBPP_PLUS=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | tail -1 | awk '{print $NF}')
    echo "MBPP pass@1: $MBPP_BASE | MBPP+ pass@1: $MBPP_PLUS"

    # LiveCodeBench
    echo "--- 3b: LiveCodeBench ---"
    python scripts/lcb_generate.py \
        --model_path "$CKPT" \
        --output_dir "livecodebench_results" \
        --release_version release_v1 \
        --temperature 0.2 \
        --max_new_tokens 512 \
        --n_samples 1 \
        --num_process_evaluate 16 \
        --timeout 6

    LCB_MODEL_ID=$(echo "$CKPT" | sed 's|/|--|g')
    LCB_EVAL_FILE="livecodebench_results/${LCB_MODEL_ID}_release_v1_eval.json"
    if [ -f "$LCB_EVAL_FILE" ]; then
        LCB_PASS1=$(python -c "import json; print(json.load(open('$LCB_EVAL_FILE'))[0]['pass@1'])")
    else
        LCB_PASS1=""
        echo "WARNING: LCB eval file not found: $LCB_EVAL_FILE"
    fi
    echo "LiveCodeBench pass@1: $LCB_PASS1"

    # ---- Step 4: 记录结果 ----
    echo "quality_r${ROUND},$CKPT,$STEPS_TOTAL,$HE_BASE,$HE_PLUS,$MBPP_BASE,$MBPP_PLUS,$LCB_PASS1,,$(date -Iseconds)" >> "$CSV"
    echo ""
    echo "Round $ROUND 结果: HE=$HE_BASE HE+=$HE_PLUS MBPP=$MBPP_BASE MBPP+=$MBPP_PLUS LCB=$LCB_PASS1"
    echo "========== Round $ROUND 完成（累计 $STEPS_TOTAL 步）=========="
done

echo ""
echo "===== [Exp B: Quality Filter] 实验完成 ====="
echo ""
echo "结果汇总："
column -t -s',' "$CSV"
