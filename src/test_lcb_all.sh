#!/bin/bash
#SBATCH --job-name=test_lcb_all
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=16:00:00
#SBATCH --output=test_lcb_all_%j.out

set -euo pipefail

# ---- Environment ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

OUT_DIR="$PROJECT_DIR/livecodebench_results/v2_baselines"
mkdir -p "$OUT_DIR"

run_model_lcb() {
    local MODEL_NAME="$1"
    local CONFIG="$2"
    local MODEL_PATH="$3"
    local VENV="$4"

    echo "============================================"
    echo "=== $MODEL_NAME: Activating venv $VENV ==="
    echo "============================================"
    source "$VENV/bin/activate"

    echo "  transformers=$(python -c 'import transformers; print(transformers.__version__)')"
    echo "  torch=$(python -c 'import torch; print(torch.__version__)')"
    echo ""

    echo "--- $MODEL_NAME: LiveCodeBench release_v1 (greedy) ---"
    python src/evaluate_lcb.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUT_DIR" \
        --release_version release_v1 \
        --temperature 0.0 \
        --n_samples 1 \
        --max_new_tokens 512 \
        --num_process_evaluate 16 \
        --timeout 6
    echo ""

    deactivate
}

# ---- 1. SantaCoder (venv with transformers==4.35.2) ----
run_model_lcb "santacoder" \
    "configs/santacoder.yaml" \
    "bigcode/santacoder" \
    "$PROJECT_DIR/venv"

# ---- 2. StarCoder2-3B (general venv with transformers>=4.39) ----
run_model_lcb "starcoder2" \
    "configs/starcoder2.yaml" \
    "bigcode/starcoder2-3b" \
    "$PROJECT_DIR/venvs/general"

# ---- 3. Qwen2.5-Coder-1.5B ----
run_model_lcb "qwen25" \
    "configs/qwen25.yaml" \
    "Qwen/Qwen2.5-Coder-1.5B" \
    "$PROJECT_DIR/venvs/general"

# ---- 4. Code Llama 7B ----
run_model_lcb "codellama" \
    "configs/codellama.yaml" \
    "codellama/CodeLlama-7b-hf" \
    "$PROJECT_DIR/venvs/general"

# ---- Summary ----
echo ""
echo "============================================"
echo "=== SUMMARY ==="
echo "============================================"
for MODEL in santacoder starcoder2 qwen25 codellama; do
    echo "--- $MODEL ---"
    SCORES="$(find "$OUT_DIR" -maxdepth 1 -type f -name "${MODEL}_*_release_v1_scores.json" | sort | tail -1)"
    if [ -n "$SCORES" ] && [ -f "$SCORES" ]; then
        cat "$SCORES"
        echo ""
    else
        echo "  LiveCodeBench: MISSING"
    fi
done

echo "=== All LiveCodeBench model tests done ==="
