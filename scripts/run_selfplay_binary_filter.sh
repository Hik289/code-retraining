#!/bin/bash
# Exp D: Self-Play + Binary Classifier Filter
#
# Each round: generate batches of 2000 -> score -> keep score>0 -> accumulate
# until TARGET_SAMPLES reached -> train -> evaluate.
#
# Usage: sbatch scripts/run_selfplay_binary_filter.sh

#SBATCH --job-name=sp_binary
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=72:00:00
#SBATCH --output=selfplay_results/logs/binary_filter_%j.out

set -euo pipefail

# ---- Project root ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

# ---- Environment ----
source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ---- Config ----
BASE_MODEL="bigcode/santacoder"
TOTAL_ROUNDS=20
STEPS_PER_ROUND=3000
BATCH_GEN=2000           # samples generated per inner loop iteration
TARGET_SAMPLES=5000      # accumulate this many passing samples per round
SCORE_THRESHOLD=0.0      # keep samples with score > threshold
ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"
EXP_DIR="selfplay_results/binary_filter"

# ---- Init ----
mkdir -p "$EXP_DIR/generated_data" selfplay_results/logs

CSV="$EXP_DIR/results.csv"
if [ ! -f "$CSV" ]; then
    echo "round,model_path,steps_total,humaneval_pass1,humaneval_plus_pass1,mbpp_pass1,mbpp_plus_pass1,livecodebench_pass1,timestamp" > "$CSV"
fi

# ---- Self-Play loop ----
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
    echo "========== [Exp D: Binary Filter] Round $ROUND / $TOTAL_ROUNDS (total $STEPS_TOTAL steps) =========="
    echo "Model: $CURRENT_MODEL"

    # ---- Step 1: Generate + score + filter until TARGET_SAMPLES reached ----
    DATA_FILE="$EXP_DIR/generated_data/round${ROUND}.jsonl"
    if [ -f "$DATA_FILE" ] && [ "$(wc -l < "$DATA_FILE")" -ge "$TARGET_SAMPLES" ]; then
        echo "--- Step 1: Skip (already have $(wc -l < "$DATA_FILE") samples) ---"
    else
        # Start fresh accumulation file
        > "$DATA_FILE"
        BATCH_IDX=0

        while true; do
            CURRENT_COUNT=$(wc -l < "$DATA_FILE")
            echo "--- Step 1 batch $BATCH_IDX: accumulated $CURRENT_COUNT / $TARGET_SAMPLES ---"

            if [ "$CURRENT_COUNT" -ge "$TARGET_SAMPLES" ]; then
                break
            fi

            SEED=$(( ROUND * 1000 + BATCH_IDX ))
            RAW_FILE="$EXP_DIR/generated_data/round${ROUND}_batch${BATCH_IDX}_raw.jsonl"
            SCORED_FILE="$EXP_DIR/generated_data/round${ROUND}_batch${BATCH_IDX}_scored.jsonl"
            FILTERED_FILE="$EXP_DIR/generated_data/round${ROUND}_batch${BATCH_IDX}_filtered.jsonl"

            # Step 1a: Generate
            echo "  1a: Generate $BATCH_GEN samples (seed=$SEED)"
            python scripts/generate_data.py \
                --model_path "$CURRENT_MODEL" \
                --output_file "$RAW_FILE" \
                --num_samples "$BATCH_GEN" \
                --seed "$SEED" \
                --local_dataset_path "$ARROW_CACHE" \
                --batch_size 128 \
                --prompt_tokens 1024 \
                --max_new_tokens 1024 \
                --temperature 0.8 \
                --top_p 0.95

            # Step 1b: Score with binary classifier
            echo "  1b: Score with binary classifier"
            python scripts/score_binary_classifier.py \
                --input_file "$RAW_FILE" \
                --output_file "$SCORED_FILE" \
                --model_path "$CURRENT_MODEL" \
                --batch_size 64

            # Step 1c: Filter (keep score > threshold)
            echo "  1c: Filter (score > $SCORE_THRESHOLD)"
            python scripts/filter_binary_classifier.py \
                --input_file "$SCORED_FILE" \
                --output_file "$FILTERED_FILE" \
                --threshold "$SCORE_THRESHOLD"

            # Append to accumulation file
            cat "$FILTERED_FILE" >> "$DATA_FILE"
            NEW_COUNT=$(wc -l < "$DATA_FILE")
            echo "  Accumulated: $NEW_COUNT samples"

            # Clean up batch intermediates to save disk
            rm -f "$RAW_FILE" "$SCORED_FILE" "$FILTERED_FILE"

            BATCH_IDX=$((BATCH_IDX + 1))
        done

        # Trim to exactly TARGET_SAMPLES
        head -n "$TARGET_SAMPLES" "$DATA_FILE" > "${DATA_FILE}.tmp" && mv "${DATA_FILE}.tmp" "$DATA_FILE"
        echo "Final dataset: $(wc -l < "$DATA_FILE") samples -> $DATA_FILE"
    fi

    LINES=$(wc -l < "$DATA_FILE")
    echo "Data: $LINES samples (binary classifier filtered)"

    # ---- Step 2: Train ----
    CKPT="$EXP_DIR/round${ROUND}/final_checkpoint"
    if [ -d "$CKPT" ] && [ -f "$CKPT/model.safetensors" ]; then
        echo "--- Step 2: Skip (checkpoint already exists) ---"
    else
        echo "--- Step 2: Train $STEPS_PER_ROUND steps ---"
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
        echo "ERROR: checkpoint $CKPT not found, aborting"
        exit 1
    fi

    # ---- Step 3: Evaluate ----
    echo "--- Step 3: Evaluate ---"

    # EvalPlus
    echo "--- 3a: EvalPlus ---"
    mkdir -p evalplus_results/humaneval evalplus_results/mbpp

    python scripts/evalplus_generate.py \
        --model_path "$CKPT" \
        --dataset humaneval \
        --output_file "evalplus_results/humaneval/binary_r${ROUND}_greedy.jsonl" \
        --temperature 0.2

    rm -f "evalplus_results/humaneval/binary_r${ROUND}_greedy_eval_results.json"
    EVAL_OUT=$(evalplus.evaluate \
        --dataset humaneval \
        --samples "evalplus_results/humaneval/binary_r${ROUND}_greedy.jsonl" \
        --i-just-wanna-run 2>&1 | tee /dev/stderr)
    HE_BASE=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | head -1 | awk '{print $NF}')
    HE_PLUS=$(echo "$EVAL_OUT" | grep -A0 'pass@1' | tail -1 | awk '{print $NF}')
    echo "HumanEval pass@1: $HE_BASE | HumanEval+ pass@1: $HE_PLUS"

    python scripts/evalplus_generate.py \
        --model_path "$CKPT" \
        --dataset mbpp \
        --output_file "evalplus_results/mbpp/binary_r${ROUND}_greedy.jsonl" \
        --temperature 0.2

    rm -f "evalplus_results/mbpp/binary_r${ROUND}_greedy_eval_results.json"
    EVAL_OUT=$(evalplus.evaluate \
        --dataset mbpp \
        --samples "evalplus_results/mbpp/binary_r${ROUND}_greedy.jsonl" \
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

    # ---- Step 4: Record results ----
    echo "binary_r${ROUND},$CKPT,$STEPS_TOTAL,$HE_BASE,$HE_PLUS,$MBPP_BASE,$MBPP_PLUS,$LCB_PASS1,$(date -Iseconds)" >> "$CSV"
    echo ""
    echo "Round $ROUND results: HE=$HE_BASE HE+=$HE_PLUS MBPP=$MBPP_BASE MBPP+=$MBPP_PLUS LCB=$LCB_PASS1"
    echo "========== Round $ROUND done (total $STEPS_TOTAL steps) =========="
done

echo ""
echo "===== [Exp D: Binary Filter] Complete ====="
echo ""
echo "Results:"
column -t -s',' "$CSV"
