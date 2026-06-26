#!/bin/bash
#SBATCH --job-name=v2_probe
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=08:00:00
#SBATCH --output=v2_probe_%x_%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch src/probe_batch_size.sh MODEL [BATCH_SIZE ...]"
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
        DEFAULT_BATCH_SIZES=(256 512 1024)
        ;;
    starcoder2)
        CONFIG="configs/starcoder2.yaml"
        MODEL_PATH="bigcode/starcoder2-3b"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(256 512 1024)
        ;;
    qwen25)
        CONFIG="configs/qwen25.yaml"
        MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(256 512 1024)
        ;;
    codellama)
        CONFIG="configs/codellama.yaml"
        MODEL_PATH="codellama/CodeLlama-7b-hf"
        VENV="$PROJECT_DIR/venvs/general"
        DEFAULT_BATCH_SIZES=(16 32 64 128)
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
export HF_DATASETS_CACHE="/tmp/self_replay_hf_datasets_cache_${SLURM_JOB_ID}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$HF_DATASETS_CACHE" "$PROJECT_DIR/results/probes"

echo "=== Probe generation batch size ==="
echo "model=$MODEL"
echo "config=$CONFIG"
echo "model_path=$MODEL_PATH"
echo "batch_sizes=${BATCH_SIZES[*]}"
echo "job_id=${SLURM_JOB_ID:-}"
echo ""

python src/probe_gen_batch_size.py \
    --config "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --num-batches 1 \
    --output "results/probes/${MODEL}_probe_${SLURM_JOB_ID:-manual}.json"

echo ""
echo "Probe done: results/probes/${MODEL}_probe_${SLURM_JOB_ID:-manual}.json"
