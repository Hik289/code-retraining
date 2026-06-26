#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --time=00:20:00
#SBATCH --output=test_train_%j.out

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

# ---- Step 1: Generate small training data ----
echo "=== Step 1: Generate 20 training samples ==="
python src/generate.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --output_file /tmp/test_train_data.jsonl \
    --num_samples 20 \
    --batch_size 4 \
    --seed 42

echo "Generated: $(wc -l < /tmp/test_train_data.jsonl) lines"
echo ""

# ---- Step 2: Train for 50 steps ----
echo "=== Step 2: Train 50 steps ==="
python src/train.py \
    --config configs/santacoder.yaml \
    --model_path bigcode/santacoder \
    --local_data_path /tmp/test_train_data.jsonl \
    --output_dir /tmp/test_train_output \
    --max_steps 50 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 10 \
    --log_freq 10 \
    --save_freq 50 \
    --seed 42
echo ""

# ---- Step 3: Verify checkpoint ----
echo "=== Step 3: Verify checkpoint ==="
CKPT="/tmp/test_train_output/final_checkpoint"
if [ -d "$CKPT" ]; then
    echo "Checkpoint exists: $CKPT"
    ls "$CKPT"/*.json "$CKPT"/*.bin 2>/dev/null || ls "$CKPT"/*.json "$CKPT"/*.safetensors 2>/dev/null || echo "  (listing files)"
    ls -lh "$CKPT/" | head -10
    # Verify generation_config has use_cache=true
    python -c "import json; gc=json.load(open('$CKPT/generation_config.json')); print(f'  use_cache={gc.get(\"use_cache\", \"MISSING\")}')"
else
    echo "ERROR: Checkpoint not found!"
    exit 1
fi
echo ""

echo "=== All training tests done ==="
