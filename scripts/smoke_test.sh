#!/bin/bash
# 冒烟测试：短训练 10 步 → 从 checkpoint 加载推理 → EvalPlus 评估
# 验证 tokenizer 保存/加载链路正确
# 用法：sbatch scripts/smoke_test.sh

#SBATCH --job-name=smoke_test
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --output=selfplay_results/logs/smoke_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled

BASE_MODEL="bigcode/santacoder"
SMOKE_DIR="selfplay_results/smoke_test"
CKPT="$SMOKE_DIR/final_checkpoint"
ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

echo "===== Step 1: 训练 10 步 ====="
torchrun --nproc_per_node 2 --standalone train.py \
    --local_arrow_path "$ARROW_CACHE" \
    --data_column content \
    --model_path "$BASE_MODEL" \
    --seq_length 2048 \
    --max_steps 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 2 \
    --weight_decay 0.05 \
    --fim_rate 0.5 \
    --fim_spm_rate 0.5 \
    --bf16 \
    --output_dir "$SMOKE_DIR" \
    --eval_freq 99999 \
    --save_freq 10 \
    --log_freq 5

echo ""
echo "===== Step 2: 检查 checkpoint ====="
ls -la "$CKPT/"
if [ -f "$CKPT/tokenizer.json" ] || [ -f "$CKPT/tokenizer_config.json" ]; then
    echo "PASS: tokenizer 文件已保存"
else
    echo "FAIL: tokenizer 文件缺失！"
    exit 1
fi

echo ""
echo "===== Step 3: 从 checkpoint 加载 → HumanEval 生成 ====="
python scripts/evalplus_generate.py \
    --model_path "$CKPT" \
    --dataset humaneval \
    --output_file "$SMOKE_DIR/humaneval_samples.jsonl" \
    --temperature 0.2

echo "PASS: 生成 $(wc -l < "$SMOKE_DIR/humaneval_samples.jsonl") 条"

echo ""
echo "===== Step 4: EvalPlus 评估 ====="
rm -f "$SMOKE_DIR/humaneval_samples_eval_results.json"
evalplus.evaluate \
    --dataset humaneval \
    --samples "$SMOKE_DIR/humaneval_samples.jsonl" \
    --i-just-wanna-run

echo ""
echo "===== Step 5: 从 checkpoint 生成数据（模拟下一轮数据生成） ====="
python scripts/generate_data.py \
    --model_path "$CKPT" \
    --output_file "$SMOKE_DIR/gen_data_test.jsonl" \
    --num_samples 5 \
    --seed 42 \
    --local_dataset_path "$ARROW_CACHE" \
    --batch_size 8 \
    --prompt_tokens 1024 \
    --max_new_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95

echo "PASS: 数据生成 $(wc -l < "$SMOKE_DIR/gen_data_test.jsonl") 条"

echo ""
echo "=========================================="
echo "  冒烟测试全部通过！"
echo "=========================================="
