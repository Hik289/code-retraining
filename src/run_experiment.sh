#!/bin/bash
#SBATCH --job-name=v2_exp
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=144:00:00
#SBATCH --output=v2_exp_%x_%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch src/run_experiment.sh MODEL FILTER [options]

MODEL:
  santacoder | starcoder2 | qwen25 | codellama

FILTER:
  none | compile | quality | ppl | binary

Options:
  --rounds N                 Number of self-play rounds (default: 5)
  --num-samples N            Final training samples per round (default: 5000)
  --raw-num-samples N        Raw samples for ppl/binary before top-k (default: num_samples*4)
  --max-steps N              Training steps per round (default: 3000)
  --output-root DIR          Result root (default: results)
  --prompt-tokens N          Generation prompt tokens (default: 1024)
  --gen-max-new-tokens N     Self-play generation tokens (default: 1024)
  --gen-batch-size N         Self-play generation batch size (default: model-specific)
  --train-seq-length N       Training sequence length (default: 2048)
  --train-batch-size N       Per-device training batch size (default: model-specific)
  --grad-accum N             Gradient accumulation steps (default: model-specific; effective train batch kept at 64)
  --warmup-steps N           Warmup on first round only (default: 500)
  --eval-limit N             EvalPlus first N tasks only (smoke test)
  --skip-evalplus            Skip official EvalPlus scoring after generation
  --lcb-debug                LiveCodeBench first 15 tasks only (smoke test)
  --lcb-max-new-tokens N     LiveCodeBench generation tokens (default: 512)
  --lcb-num-process-evaluate N
                              LiveCodeBench evaluator workers (default: model-specific)
  --skip-eval                Skip EvalPlus and LiveCodeBench
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

MODEL="$1"
FILTER="$2"
shift 2

ROUNDS=5
NUM_SAMPLES=5000
RAW_NUM_SAMPLES=""
MAX_STEPS=3000
OUTPUT_ROOT="results"
PROMPT_TOKENS=1024
GEN_MAX_NEW_TOKENS=1024
GEN_BATCH_SIZE=""
TRAIN_SEQ_LENGTH=2048
TRAIN_BATCH_SIZE=""
GRAD_ACCUM=""
WARMUP_STEPS=500
EVAL_LIMIT=""
LCB_DEBUG=0
LCB_MAX_NEW_TOKENS=512
LCB_NUM_PROCESS_EVALUATE=""
SKIP_EVAL=0
SKIP_EVALPLUS=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --raw-num-samples) RAW_NUM_SAMPLES="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --gen-max-new-tokens) GEN_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gen-batch-size) GEN_BATCH_SIZE="$2"; shift 2 ;;
        --train-seq-length) TRAIN_SEQ_LENGTH="$2"; shift 2 ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
        --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
        --skip-evalplus) SKIP_EVALPLUS=1; shift ;;
        --lcb-debug) LCB_DEBUG=1; shift ;;
        --lcb-max-new-tokens) LCB_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --lcb-num-process-evaluate) LCB_NUM_PROCESS_EVALUATE="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

case "$MODEL" in
    santacoder)
        CONFIG="configs/santacoder.yaml"
        BASE_MODEL="bigcode/santacoder"
        VENV="$PROJECT_DIR/venv"
        DEFAULT_GEN_BATCH_SIZE=512
        DEFAULT_TRAIN_BATCH_SIZE=32
        DEFAULT_GRAD_ACCUM=2
        DEFAULT_LCB_NUM_PROCESS_EVALUATE=16
        ;;
    starcoder2)
        CONFIG="configs/starcoder2.yaml"
        BASE_MODEL="bigcode/starcoder2-3b"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_GEN_BATCH_SIZE=512
        DEFAULT_TRAIN_BATCH_SIZE=8
        DEFAULT_GRAD_ACCUM=8
        DEFAULT_LCB_NUM_PROCESS_EVALUATE=16
        ;;
    qwen25)
        CONFIG="configs/qwen25.yaml"
        BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_GEN_BATCH_SIZE=512
        DEFAULT_TRAIN_BATCH_SIZE=16
        DEFAULT_GRAD_ACCUM=4
        DEFAULT_LCB_NUM_PROCESS_EVALUATE=16
        ;;
    codellama)
        CONFIG="configs/codellama.yaml"
        BASE_MODEL="codellama/CodeLlama-7b-hf"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_GEN_BATCH_SIZE=64
        DEFAULT_TRAIN_BATCH_SIZE=8
        DEFAULT_GRAD_ACCUM=8
        DEFAULT_LCB_NUM_PROCESS_EVALUATE=1
        ;;
    *)
        echo "ERROR: unknown model '$MODEL'"
        usage
        exit 1
        ;;
esac

if [ -z "$GEN_BATCH_SIZE" ]; then
    GEN_BATCH_SIZE="$DEFAULT_GEN_BATCH_SIZE"
fi

if [ -z "$TRAIN_BATCH_SIZE" ]; then
    TRAIN_BATCH_SIZE="$DEFAULT_TRAIN_BATCH_SIZE"
fi

if [ -z "$GRAD_ACCUM" ]; then
    GRAD_ACCUM="$DEFAULT_GRAD_ACCUM"
fi

if [ -z "$LCB_NUM_PROCESS_EVALUATE" ]; then
    LCB_NUM_PROCESS_EVALUATE="$DEFAULT_LCB_NUM_PROCESS_EVALUATE"
fi

case "$FILTER" in
    none|compile|quality|ppl|binary) ;;
    *)
        echo "ERROR: unknown filter '$FILTER'"
        usage
        exit 1
        ;;
esac

if [ -z "$RAW_NUM_SAMPLES" ]; then
    RAW_NUM_SAMPLES=$((NUM_SAMPLES * 4))
fi

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_DIR="$PROJECT_DIR/$OUTPUT_ROOT/$MODEL/$FILTER"
DATA_DIR="$EXP_DIR/generated_data"
EVALPLUS_DIR="$EXP_DIR/evalplus"
LCB_DIR="$EXP_DIR/livecodebench"
mkdir -p "$DATA_DIR" "$EVALPLUS_DIR" "$LCB_DIR"

CSV="$EXP_DIR/results.csv"
if [ ! -f "$CSV" ]; then
    echo "model,filter,round,steps_total,humaneval_pass1,humaneval_plus_pass1,mbpp_pass1,mbpp_plus_pass1,livecodebench_pass1,train_loss,num_generated,num_after_filter,filter_pass_rate,generation_time_sec,training_time_sec,eval_time_sec,timestamp" > "$CSV"
fi

echo "=== V2 experiment ==="
echo "model=$MODEL filter=$FILTER rounds=$ROUNDS samples=$NUM_SAMPLES steps=$MAX_STEPS"
echo "config=$CONFIG base_model=$BASE_MODEL venv=$VENV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

json_get() {
    local path="$1"
    local key="$2"
    python -c "import json,sys; p,k=sys.argv[1:3]; print(json.load(open(p)).get(k,''))" "$path" "$key"
}

json_lines() {
    local path="$1"
    if [ -f "$path" ]; then
        wc -l < "$path"
    else
        echo 0
    fi
}

generation_complete() {
    local filter="$1"
    local data_file="$2"
    local raw_file="$3"
    local scored_file="$4"

    if [ "$filter" = "ppl" ] || [ "$filter" = "binary" ]; then
        [ "$(json_lines "$raw_file")" -ge "$RAW_NUM_SAMPLES" ] \
            && [ "$(json_lines "$scored_file")" -ge "$RAW_NUM_SAMPLES" ] \
            && [ "$(json_lines "$data_file")" -gt 0 ]
    else
        [ "$(json_lines "$data_file")" -ge "$NUM_SAMPLES" ]
    fi
}

csv_has_round() {
    local path="$1"
    local round="$2"
    python - "$path" "$round" <<'PY'
import csv
import os
import sys

path, wanted = sys.argv[1:3]
if not os.path.isfile(path):
    sys.exit(1)

with open(path, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("round") == wanted:
            sys.exit(0)
sys.exit(1)
PY
}

lcb_scores_for_round() {
    local round="$1"
    find "$LCB_DIR" -maxdepth 1 -type f \
        -name "${MODEL}_*--round${round}--final_checkpoint_release_v1*_scores.json" \
        | sort | tail -1
}

for ROUND in $(seq 1 "$ROUNDS"); do
    if [ "$ROUND" -eq 1 ]; then
        CURRENT_MODEL="$BASE_MODEL"
        ROUND_WARMUP="$WARMUP_STEPS"
    else
        CURRENT_MODEL="$EXP_DIR/round$((ROUND - 1))/final_checkpoint"
        ROUND_WARMUP=0
    fi

    STEPS_TOTAL=$((ROUND * MAX_STEPS))
    ROUND_DIR="$EXP_DIR/round${ROUND}"
    CKPT="$ROUND_DIR/final_checkpoint"
    DATA_FILE="$DATA_DIR/round${ROUND}.jsonl"
    RAW_FILE="$DATA_DIR/round${ROUND}_raw.jsonl"
    SCORED_FILE="$DATA_DIR/round${ROUND}_scored.jsonl"
    META_FILE="$EXP_DIR/round${ROUND}_meta.json"

    echo ""
    echo "========== Round $ROUND / $ROUNDS =========="
    echo "current_model=$CURRENT_MODEL"

    if csv_has_round "$CSV" "$ROUND"; then
        echo "--- Round $ROUND: skip existing CSV row"
        continue
    fi

    GEN_START=$(date +%s)
    if generation_complete "$FILTER" "$DATA_FILE" "$RAW_FILE" "$SCORED_FILE"; then
        echo "--- Generation: skip existing $DATA_FILE ($(json_lines "$DATA_FILE") lines)"
    else
        case "$FILTER" in
            none)
                python src/generate.py \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --output_file "$DATA_FILE" \
                    --num_samples "$NUM_SAMPLES" \
                    --seed "$ROUND" \
                    --prompt_tokens "$PROMPT_TOKENS" \
                    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
                    --batch_size "$GEN_BATCH_SIZE"
                ;;
            compile)
                python src/generate.py \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --output_file "$DATA_FILE" \
                    --num_samples "$NUM_SAMPLES" \
                    --seed "$ROUND" \
                    --prompt_tokens "$PROMPT_TOKENS" \
                    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
                    --batch_size "$GEN_BATCH_SIZE" \
                    --filter_mode compile
                ;;
            quality)
                python src/generate.py \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --output_file "$DATA_FILE" \
                    --num_samples "$NUM_SAMPLES" \
                    --seed "$ROUND" \
                    --prompt_tokens "$PROMPT_TOKENS" \
                    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
                    --batch_size "$GEN_BATCH_SIZE" \
                    --filter_mode compile+quality
                ;;
            ppl)
                python src/generate.py \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --output_file "$RAW_FILE" \
                    --num_samples "$RAW_NUM_SAMPLES" \
                    --seed "$ROUND" \
                    --prompt_tokens "$PROMPT_TOKENS" \
                    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
                    --batch_size "$GEN_BATCH_SIZE"
                python src/filters.py score-ppl \
                    --input_file "$RAW_FILE" \
                    --output_file "$SCORED_FILE" \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --prompt_tokens "$PROMPT_TOKENS"
                python src/filters.py filter-topk \
                    --input_file "$SCORED_FILE" \
                    --output_file "$DATA_FILE" \
                    --score_field ppl \
                    --top_percent 25 \
                    --ascending
                ;;
            binary)
                python src/generate.py \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL" \
                    --output_file "$RAW_FILE" \
                    --num_samples "$RAW_NUM_SAMPLES" \
                    --seed "$ROUND" \
                    --prompt_tokens "$PROMPT_TOKENS" \
                    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
                    --batch_size "$GEN_BATCH_SIZE"
                python src/filters.py score-binary \
                    --input_file "$RAW_FILE" \
                    --output_file "$SCORED_FILE" \
                    --config "$CONFIG" \
                    --model_path "$CURRENT_MODEL"
                python src/filters.py filter-topk \
                    --input_file "$SCORED_FILE" \
                    --output_file "$DATA_FILE" \
                    --score_field score \
                    --top_percent 25
                ;;
        esac
    fi
    GEN_END=$(date +%s)
    GENERATION_TIME=$((GEN_END - GEN_START))

    NUM_AFTER_FILTER=$(json_lines "$DATA_FILE")
    if [ "$FILTER" = "ppl" ] || [ "$FILTER" = "binary" ]; then
        NUM_GENERATED=$(json_lines "$RAW_FILE")
    else
        NUM_GENERATED="$NUM_AFTER_FILTER"
    fi
    FILTER_PASS_RATE=$(python -c "n=float('$NUM_GENERATED'); k=float('$NUM_AFTER_FILTER'); print(round(k/n, 4) if n else 0.0)")
    echo "data=$DATA_FILE generated=$NUM_GENERATED after_filter=$NUM_AFTER_FILTER pass_rate=$FILTER_PASS_RATE"

    TRAIN_START=$(date +%s)
    if [ -d "$CKPT" ] && { [ -f "$CKPT/model.safetensors" ] || ls "$CKPT"/model-*.safetensors >/dev/null 2>&1; }; then
        echo "--- Training: skip existing $CKPT"
    else
        python src/train.py \
            --config "$CONFIG" \
            --model_path "$CURRENT_MODEL" \
            --local_data_path "$DATA_FILE" \
            --seq_length "$TRAIN_SEQ_LENGTH" \
            --max_steps "$MAX_STEPS" \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --warmup_steps "$ROUND_WARMUP" \
            --output_dir "$ROUND_DIR" \
            --save_freq "$MAX_STEPS" \
            --eval_freq "$MAX_STEPS" \
            --log_freq 10
    fi
    TRAIN_END=$(date +%s)
    TRAINING_TIME=$((TRAIN_END - TRAIN_START))

    if [ ! -d "$CKPT" ]; then
        echo "ERROR: checkpoint missing: $CKPT"
        exit 1
    fi

    HE_BASE=""
    HE_PLUS=""
    MBPP_BASE=""
    MBPP_PLUS=""
    LCB_PASS1=""
    EVAL_START=$(date +%s)
    if [ "$SKIP_EVAL" -eq 1 ]; then
        echo "--- Evaluation: skipped"
    else
        EVAL_LIMIT_ARGS=()
        if [ -n "$EVAL_LIMIT" ]; then
            EVAL_LIMIT_ARGS=(--limit "$EVAL_LIMIT")
        fi
        EVALPLUS_SKIP_ARGS=()
        if [ "$SKIP_EVALPLUS" -eq 1 ]; then
            EVALPLUS_SKIP_ARGS=(--skip_evaluate)
        fi

        HE_SCORES="$EVALPLUS_DIR/round${ROUND}_humaneval_scores.json"
        if [ -f "$HE_SCORES" ]; then
            echo "--- EvalPlus HumanEval: skip existing $HE_SCORES"
        else
            python src/evaluate_evalplus.py \
                --config "$CONFIG" \
                --model_path "$CKPT" \
                --dataset humaneval \
                --output_file "$EVALPLUS_DIR/round${ROUND}_humaneval.jsonl" \
                --temperature 0.0 \
                --n_samples 1 \
                "${EVAL_LIMIT_ARGS[@]}" \
                "${EVALPLUS_SKIP_ARGS[@]}"
        fi
        HE_BASE=$(json_get "$HE_SCORES" humaneval_pass1)
        HE_PLUS=$(json_get "$HE_SCORES" humaneval_plus_pass1)

        MBPP_SCORES="$EVALPLUS_DIR/round${ROUND}_mbpp_scores.json"
        if [ -f "$MBPP_SCORES" ]; then
            echo "--- EvalPlus MBPP: skip existing $MBPP_SCORES"
        else
            python src/evaluate_evalplus.py \
                --config "$CONFIG" \
                --model_path "$CKPT" \
                --dataset mbpp \
                --output_file "$EVALPLUS_DIR/round${ROUND}_mbpp.jsonl" \
                --temperature 0.0 \
                --n_samples 1 \
                "${EVAL_LIMIT_ARGS[@]}" \
                "${EVALPLUS_SKIP_ARGS[@]}"
        fi
        MBPP_BASE=$(json_get "$MBPP_SCORES" mbpp_pass1)
        MBPP_PLUS=$(json_get "$MBPP_SCORES" mbpp_plus_pass1)

        LCB_ARGS=()
        if [ "$LCB_DEBUG" -eq 1 ]; then
            LCB_ARGS=(--debug)
        fi
        LCB_SCORES="$(lcb_scores_for_round "$ROUND")"
        if [ -n "$LCB_SCORES" ] && [ -f "$LCB_SCORES" ]; then
            echo "--- LiveCodeBench: skip existing $LCB_SCORES"
        else
            python src/evaluate_lcb.py \
                --config "$CONFIG" \
                --model_path "$CKPT" \
                --output_dir "$LCB_DIR" \
                --release_version release_v1 \
                --temperature 0.0 \
                --n_samples 1 \
                --max_new_tokens "$LCB_MAX_NEW_TOKENS" \
                --num_process_evaluate "$LCB_NUM_PROCESS_EVALUATE" \
                --timeout 6 \
                "${LCB_ARGS[@]}"
            LCB_SCORES="$(lcb_scores_for_round "$ROUND")"
        fi
        if [ -n "$LCB_SCORES" ]; then
            LCB_PASS1=$(json_get "$LCB_SCORES" livecodebench_pass1)
        fi
    fi
    EVAL_END=$(date +%s)
    EVAL_TIME=$((EVAL_END - EVAL_START))

    TRAIN_LOSS=""
    TRAIN_STATE="$ROUND_DIR/trainer_state.json"
    if [ ! -f "$TRAIN_STATE" ]; then
        TRAIN_STATE="$(find "$ROUND_DIR" -maxdepth 2 -type f -path '*/checkpoint-*/trainer_state.json' | sort -V | tail -1)"
    fi
    if [ -n "$TRAIN_STATE" ] && [ -f "$TRAIN_STATE" ]; then
        TRAIN_LOSS=$(python -c "import json,sys; h=json.load(open(sys.argv[1])).get('log_history', []); vals=[e.get('loss') for e in h if 'loss' in e]; vals2=[e.get('train_loss') for e in h if 'train_loss' in e]; print((vals or vals2 or [''])[-1])" "$TRAIN_STATE")
    fi

    TIMESTAMP=$(date -Iseconds)
    echo "$MODEL,$FILTER,$ROUND,$STEPS_TOTAL,$HE_BASE,$HE_PLUS,$MBPP_BASE,$MBPP_PLUS,$LCB_PASS1,$TRAIN_LOSS,$NUM_GENERATED,$NUM_AFTER_FILTER,$FILTER_PASS_RATE,$GENERATION_TIME,$TRAINING_TIME,$EVAL_TIME,$TIMESTAMP" >> "$CSV"

    python - "$META_FILE" <<EOF
import json
meta = {
    "model": "$MODEL",
    "filter": "$FILTER",
    "round": $ROUND,
    "model_path": "$CURRENT_MODEL",
    "checkpoint": "$CKPT",
    "data_file": "$DATA_FILE",
    "steps_total": $STEPS_TOTAL,
    "metrics": {
        "humaneval_pass1": "$HE_BASE",
        "humaneval_plus_pass1": "$HE_PLUS",
        "mbpp_pass1": "$MBPP_BASE",
        "mbpp_plus_pass1": "$MBPP_PLUS",
        "livecodebench_pass1": "$LCB_PASS1",
        "train_loss": "$TRAIN_LOSS"
    },
    "counts": {
        "num_generated": $NUM_GENERATED,
        "num_after_filter": $NUM_AFTER_FILTER,
        "filter_pass_rate": $FILTER_PASS_RATE
    },
    "timing_sec": {
        "generation": $GENERATION_TIME,
        "training": $TRAINING_TIME,
        "eval": $EVAL_TIME
    },
    "timestamp": "$TIMESTAMP"
}
with open("$META_FILE", "w") as f:
    json.dump(meta, f, indent=2)
EOF

    echo "Round $ROUND done: HE=$HE_BASE HE+=$HE_PLUS MBPP=$MBPP_BASE MBPP+=$MBPP_PLUS LCB=$LCB_PASS1"
done

echo ""
echo "=== Experiment complete ==="
echo "CSV: $CSV"
