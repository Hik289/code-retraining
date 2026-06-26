#!/bin/bash
#SBATCH --job-name=v2_probe_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=08:00:00
#SBATCH --output=v2_probe_train_%x_%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch src/probe_train_batch_size.sh MODEL [BATCH_SIZE ...]"
    exit 1
fi

MODEL="$1"
shift || true

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

case "$MODEL" in
    santacoder)
        CONFIG="configs/santacoder.yaml"
        MODEL_PATH="bigcode/santacoder"
        VENV="$PROJECT_DIR/venv"
        DEFAULT_BATCH_SIZES=(8 16 32 64 128 256)
        ;;
    starcoder2)
        CONFIG="configs/starcoder2.yaml"
        MODEL_PATH="bigcode/starcoder2-3b"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(8 16 32 64 128 256)
        ;;
    qwen25)
        CONFIG="configs/qwen25.yaml"
        MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(8 16 32 64 128 256)
        ;;
    codellama)
        CONFIG="configs/codellama.yaml"
        MODEL_PATH="codellama/CodeLlama-7b-hf"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(8 16 32 64 128 256)
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

if [ "$#" -gt 0 ]; then
    BATCH_SIZES=("$@")
else
    BATCH_SIZES=("${DEFAULT_BATCH_SIZES[@]}")
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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$PROJECT_DIR/results/probes"

echo "=== Probe training batch size ==="
echo "model=$MODEL"
echo "config=$CONFIG"
echo "model_path=$MODEL_PATH"
echo "batch_sizes=${BATCH_SIZES[*]}"
echo "job_id=${SLURM_JOB_ID:-}"
if [ -f /tmp/nvidia-smi-output ]; then
    echo "--- GPU snapshot ---"
    cat /tmp/nvidia-smi-output
    echo ""
fi

python src/probe_train_batch_size.py \
    --config "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --steps 2 \
    --output "results/probes/${MODEL}_train_probe_${SLURM_JOB_ID:-manual}.json"

echo ""
echo "Train probe done: results/probes/${MODEL}_train_probe_${SLURM_JOB_ID:-manual}.json"
