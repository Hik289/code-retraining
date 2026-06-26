#!/bin/bash
#SBATCH --job-name=test_evalplus_all
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=12:00:00
#SBATCH --output=test_evalplus_all_%j.out

set -euo pipefail

# ---- Environment ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

OUT_DIR="/tmp/test_evalplus_all"
mkdir -p "$OUT_DIR"

# Helper function: run evalplus for one model (humaneval + mbpp)
run_model_eval() {
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

    echo "--- $MODEL_NAME: HumanEval (greedy) ---"
    python src/evaluate_evalplus.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --dataset humaneval \
        --output_file "$OUT_DIR/${MODEL_NAME}_humaneval.jsonl" \
        --temperature 0.0 \
        --n_samples 1
    echo ""

    echo "--- $MODEL_NAME: MBPP (greedy) ---"
    python src/evaluate_evalplus.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --dataset mbpp \
        --output_file "$OUT_DIR/${MODEL_NAME}_mbpp.jsonl" \
        --temperature 0.0 \
        --n_samples 1
    echo ""

    deactivate
}

# ---- 1. SantaCoder (venv with transformers==4.35.2) ----
run_model_eval "santacoder" \
    "configs/santacoder.yaml" \
    "bigcode/santacoder" \
    "$PROJECT_DIR/venv"

# ---- 2. StarCoder2-3B (general venv with transformers>=4.39) ----
run_model_eval "starcoder2" \
    "configs/starcoder2.yaml" \
    "bigcode/starcoder2-3b" \
    "$PROJECT_DIR/venvs/general"

# ---- 3. Qwen2.5-Coder-1.5B ----
run_model_eval "qwen25" \
    "configs/qwen25.yaml" \
    "Qwen/Qwen2.5-Coder-1.5B" \
    "$PROJECT_DIR/venvs/general"

# ---- 4. Code Llama 7B ----
run_model_eval "codellama" \
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
    for DATASET in humaneval mbpp; do
        SCORES="$OUT_DIR/${MODEL}_${DATASET}_scores.json"
        if [ -f "$SCORES" ]; then
            cat "$SCORES"
        else
            echo "  $DATASET: MISSING"
        fi
    done
    echo ""
done

echo "=== All evalplus tests done ==="
