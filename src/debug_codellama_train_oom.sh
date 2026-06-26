#!/bin/bash
#SBATCH --job-name=v2_debug_codellama_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=06:00:00
#SBATCH --output=v2_debug_codellama_train_%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch src/debug_codellama_train_oom.sh [FILTER] [BSxGA ...]

FILTER:
  none | compile | quality | ppl | binary | all
  Default: none

BSxGA:
  Candidate per-device batch size and gradient accumulation.
  Default: 8x8 4x16

Examples:
  sbatch src/debug_codellama_train_oom.sh none 8x8 4x16
  sbatch src/debug_codellama_train_oom.sh all 8x8
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

FILTER="${1:-none}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$FILTER" in
    none|compile|quality|ppl|binary|all) ;;
    *)
        echo "ERROR: unknown filter '$FILTER'"
        usage
        exit 1
        ;;
esac

if [ "$#" -gt 0 ]; then
    CANDIDATES=("$@")
else
    CANDIDATES=(8x8 4x16)
fi

CONFIG="configs/codellama.yaml"
MODEL_PATH="codellama/CodeLlama-7b-hf"
VENV="$PROJECT_DIR/venvs/general"
MAX_STEPS="${MAX_STEPS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
LOG_FREQ="${LOG_FREQ:-1}"
WARMUP_STEPS=0
SAVE_FREQ=999999
DEBUG_ROOT="$PROJECT_DIR/results/debug/codellama_train_oom/${SLURM_JOB_ID:-manual}"
SUMMARY="$DEBUG_ROOT/summary.tsv"

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
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$DEBUG_ROOT"
printf "filter\tcandidate\tbatch_size\tgrad_accum\teffective_batch\tstatus\n" > "$SUMMARY"

echo "=== CodeLlama real train OOM debug ==="
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "filter=$FILTER"
echo "candidates=${CANDIDATES[*]}"
echo "max_steps=$MAX_STEPS seq_length=$SEQ_LENGTH"
echo "debug_root=$DEBUG_ROOT"
echo "summary=$SUMMARY"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ -f /tmp/nvidia-smi-output ]; then
    echo "--- GPU snapshot ---"
    cat /tmp/nvidia-smi-output
    echo ""
fi

if [ "$FILTER" = "all" ]; then
    FILTERS=(none compile quality ppl binary)
else
    FILTERS=("$FILTER")
fi

overall_status=0

for current_filter in "${FILTERS[@]}"; do
    DATA_FILE="$PROJECT_DIR/results/codellama/$current_filter/generated_data/round1.jsonl"
    if [ ! -s "$DATA_FILE" ]; then
        echo "ERROR: missing data file: $DATA_FILE"
        printf "%s\t-\t-\t-\t-\tMISSING_DATA\n" "$current_filter" >> "$SUMMARY"
        overall_status=1
        continue
    fi

    echo ""
    echo "===== filter=$current_filter data=$(wc -l < "$DATA_FILE") lines ====="

    for candidate in "${CANDIDATES[@]}"; do
        if [[ ! "$candidate" =~ ^[0-9]+x[0-9]+$ ]]; then
            echo "ERROR: candidate must look like BSxGA, got '$candidate'"
            printf "%s\t%s\t-\t-\t-\tBAD_CANDIDATE\n" "$current_filter" "$candidate" >> "$SUMMARY"
            overall_status=1
            continue
        fi

        batch_size="${candidate%x*}"
        grad_accum="${candidate#*x}"
        effective_batch=$((batch_size * grad_accum))
        OUT_DIR="$DEBUG_ROOT/${current_filter}_${candidate}"

        echo ""
        echo "----- candidate=$candidate batch_size=$batch_size grad_accum=$grad_accum effective=$effective_batch -----"
        set +e
        python src/train.py \
            --config "$CONFIG" \
            --model_path "$MODEL_PATH" \
            --local_data_path "$DATA_FILE" \
            --seq_length "$SEQ_LENGTH" \
            --max_steps "$MAX_STEPS" \
            --batch_size "$batch_size" \
            --gradient_accumulation_steps "$grad_accum" \
            --warmup_steps "$WARMUP_STEPS" \
            --output_dir "$OUT_DIR" \
            --save_freq "$SAVE_FREQ" \
            --eval_freq "$SAVE_FREQ" \
            --log_freq "$LOG_FREQ" \
            --skip_final_save
        rc=$?
        set -e

        if [ "$rc" -eq 0 ]; then
            status="OK"
            echo "RESULT $current_filter $candidate: OK"
        else
            status="FAIL_$rc"
            overall_status=1
            echo "RESULT $current_filter $candidate: FAIL rc=$rc"
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$current_filter" "$candidate" "$batch_size" "$grad_accum" "$effective_batch" "$status" >> "$SUMMARY"

        python - <<'PY'
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
PY
    done
done

echo ""
echo "=== Summary ==="
cat "$SUMMARY"

exit "$overall_status"
