#!/bin/bash
#SBATCH --job-name=test_evalplus
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --output=test_evalplus_%j.out

set -euo pipefail

# ---- Environment ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
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

# ---- Step 1: HumanEval with SantaCoder (base model, greedy) ----
echo "=== Step 1: HumanEval evaluation (SantaCoder, greedy) ==="
python src/evaluate_evalplus.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --dataset humaneval \
    --output_file /tmp/test_evalplus/humaneval_santacoder.jsonl \
    --temperature 0.0 \
    --n_samples 1

echo ""

# ---- Step 2: MBPP with SantaCoder (base model, greedy) ----
echo "=== Step 2: MBPP evaluation (SantaCoder, greedy) ==="
python src/evaluate_evalplus.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --dataset mbpp \
    --output_file /tmp/test_evalplus/mbpp_santacoder.jsonl \
    --temperature 0.0 \
    --n_samples 1

echo ""

# ---- Step 3: Verify output files ----
echo "=== Step 3: Verify outputs ==="
echo "HumanEval samples: $(wc -l < /tmp/test_evalplus/humaneval_santacoder.jsonl)"
echo "MBPP samples: $(wc -l < /tmp/test_evalplus/mbpp_santacoder.jsonl)"

if [ -f /tmp/test_evalplus/humaneval_santacoder_scores.json ]; then
    echo "HumanEval scores:"
    cat /tmp/test_evalplus/humaneval_santacoder_scores.json
else
    echo "WARNING: HumanEval scores file not found"
fi
echo ""

if [ -f /tmp/test_evalplus/mbpp_santacoder_scores.json ]; then
    echo "MBPP scores:"
    cat /tmp/test_evalplus/mbpp_santacoder_scores.json
else
    echo "WARNING: MBPP scores file not found"
fi
echo ""

echo "=== All evalplus tests done ==="
