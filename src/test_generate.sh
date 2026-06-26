#!/bin/bash
#SBATCH --job-name=test_gen
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=00:15:00
#SBATCH --output=test_generate_%j.out

set -euo pipefail

# ---- 环境 ----
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- Test 1: no filter, 10 samples ----
echo "=== Test 1: generate 10 samples, no filter ==="
python src/generate.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --output_file /tmp/test_gen_nofilter.jsonl \
    --num_samples 10 \
    --batch_size 2 \
    --seed 42

echo ""
echo "--- Output check (no filter) ---"
echo "Lines: $(wc -l < /tmp/test_gen_nofilter.jsonl)"
echo "First record:"
head -1 /tmp/test_gen_nofilter.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(f'  content length: {len(d[\"content\"])} chars'); print(f'  first 100 chars: {d[\"content\"][:100]}')"

# ---- Test 2: compile filter, 10 samples ----
echo ""
echo "=== Test 2: generate 10 samples, compile filter ==="
python src/generate.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --output_file /tmp/test_gen_compile.jsonl \
    --num_samples 10 \
    --batch_size 4 \
    --seed 42 \
    --filter_mode compile

echo ""
echo "--- Output check (compile) ---"
echo "Lines: $(wc -l < /tmp/test_gen_compile.jsonl)"
echo "First record:"
head -1 /tmp/test_gen_compile.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(f'  content length: {len(d[\"content\"])} chars'); print(f'  first 100 chars: {d[\"content\"][:100]}')"

echo ""
echo "=== All tests done ==="
