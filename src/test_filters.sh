#!/bin/bash
#SBATCH --job-name=test_filters
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=00:15:00
#SBATCH --output=test_filters_%j.out

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

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- Step 0: Generate small test data (10 samples, no filter) ----
echo "=== Step 0: Generate 10 test samples ==="
python src/generate.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --output_file /tmp/test_filter_raw.jsonl \
    --num_samples 10 \
    --batch_size 4 \
    --seed 42

echo "Generated: $(wc -l < /tmp/test_filter_raw.jsonl) lines"
echo ""

# ---- Step 1: compile filter (CPU) ----
echo "=== Step 1: compile filter ==="
python src/filters.py compile \
    --input_file /tmp/test_filter_raw.jsonl \
    --output_file /tmp/test_filter_compile.jsonl
echo ""

# ---- Step 2: quality filter (CPU, needs tokenizer) ----
echo "=== Step 2: quality filter ==="
python src/filters.py quality \
    --input_file /tmp/test_filter_raw.jsonl \
    --output_file /tmp/test_filter_quality.jsonl \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder
echo ""

# ---- Step 3: score-ppl (GPU) ----
echo "=== Step 3: PPL scoring ==="
python src/filters.py score-ppl \
    --input_file /tmp/test_filter_raw.jsonl \
    --output_file /tmp/test_filter_ppl_scored.jsonl \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --batch_size 4
echo ""
echo "--- PPL scored sample ---"
head -1 /tmp/test_filter_ppl_scored.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(f'  ppl={d[\"ppl\"]:.2f}, content_len={len(d[\"content\"])}')"
echo ""

# ---- Step 4: filter-topk on PPL (CPU) ----
echo "=== Step 4: filter-topk PPL (top 50%, ascending) ==="
python src/filters.py filter-topk \
    --input_file /tmp/test_filter_ppl_scored.jsonl \
    --output_file /tmp/test_filter_ppl_filtered.jsonl \
    --score_field ppl \
    --top_percent 50 \
    --ascending
echo ""

# ---- Step 5: score-binary (GPU) ----
echo "=== Step 5: Binary scoring ==="
python src/filters.py score-binary \
    --input_file /tmp/test_filter_raw.jsonl \
    --output_file /tmp/test_filter_binary_scored.jsonl \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --batch_size 4
echo ""
echo "--- Binary scored sample ---"
head -1 /tmp/test_filter_binary_scored.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(f'  score={d[\"score\"]:.4f}, content_len={len(d[\"content\"])}')"
echo ""

# ---- Step 6: filter-topk on binary score (CPU) ----
echo "=== Step 6: filter-topk binary (top 50%, descending) ==="
python src/filters.py filter-topk \
    --input_file /tmp/test_filter_binary_scored.jsonl \
    --output_file /tmp/test_filter_binary_filtered.jsonl \
    --score_field score \
    --top_percent 50
echo ""

# ---- Summary ----
echo "=== Summary ==="
echo "Raw:              $(wc -l < /tmp/test_filter_raw.jsonl) samples"
echo "After compile:    $(wc -l < /tmp/test_filter_compile.jsonl) samples"
echo "After quality:    $(wc -l < /tmp/test_filter_quality.jsonl) samples"
echo "PPL scored:       $(wc -l < /tmp/test_filter_ppl_scored.jsonl) samples"
echo "PPL filtered:     $(wc -l < /tmp/test_filter_ppl_filtered.jsonl) samples"
echo "Binary scored:    $(wc -l < /tmp/test_filter_binary_scored.jsonl) samples"
echo "Binary filtered:  $(wc -l < /tmp/test_filter_binary_filtered.jsonl) samples"
echo ""
echo "=== All filter tests done ==="
