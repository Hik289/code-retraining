#!/bin/bash
# Smoke test: 验证三个过滤实验的脚本能跑通
# 用法：sbatch scripts/smoke_test_filters.sh
# 需要 GPU，生成极少量样本快速验证

#SBATCH --job-name=smoke_filter
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --output=selfplay_results/logs/smoke_filter_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

BASE_MODEL="bigcode/santacoder"
ARROW_CACHE="${THE_STACK_ARROW_CACHE:-}"
TEST_DIR="selfplay_results/smoke_filter"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR" selfplay_results/logs

PASS=0
FAIL=0

# ========== Exp A: compile filter ==========
echo ""
echo "===== [Test 1/4] Exp A: generate_data_filtered.py --filter_mode compile ====="
OUT_A="$TEST_DIR/expA.jsonl"
if python scripts/generate_data_filtered.py \
    --model_path "$BASE_MODEL" \
    --output_file "$OUT_A" \
    --num_samples 10 \
    --seed 42 \
    --local_dataset_path "$ARROW_CACHE" \
    --batch_size 16 \
    --prompt_tokens 512 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --filter_mode compile; then
    LINES_A=$(wc -l < "$OUT_A")
    echo "PASS: Exp A 生成 $LINES_A 条 (目标 10)"
    PASS=$((PASS + 1))
else
    echo "FAIL: Exp A 报错"
    FAIL=$((FAIL + 1))
fi

# ========== Exp B: compile+quality filter ==========
echo ""
echo "===== [Test 2/4] Exp B: generate_data_filtered.py --filter_mode compile+quality ====="
OUT_B="$TEST_DIR/expB.jsonl"
if python scripts/generate_data_filtered.py \
    --model_path "$BASE_MODEL" \
    --output_file "$OUT_B" \
    --num_samples 10 \
    --seed 42 \
    --local_dataset_path "$ARROW_CACHE" \
    --batch_size 16 \
    --prompt_tokens 512 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --filter_mode compile+quality \
    --repetition_threshold 0.5 \
    --min_completion_tokens 50; then
    LINES_B=$(wc -l < "$OUT_B")
    echo "PASS: Exp B 生成 $LINES_B 条 (目标 10)"
    PASS=$((PASS + 1))
else
    echo "FAIL: Exp B 报错"
    FAIL=$((FAIL + 1))
fi

# ========== Exp C: PPL filter (3 steps) ==========

# Step C1: 用 generate_data.py 生成原始数据
echo ""
echo "===== [Test 3/4] Exp C Step 1: generate_data.py (无过滤) ====="
OUT_C_RAW="$TEST_DIR/expC_raw.jsonl"
if python scripts/generate_data.py \
    --model_path "$BASE_MODEL" \
    --output_file "$OUT_C_RAW" \
    --num_samples 20 \
    --seed 42 \
    --local_dataset_path "$ARROW_CACHE" \
    --batch_size 16 \
    --prompt_tokens 512 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95; then
    LINES_RAW=$(wc -l < "$OUT_C_RAW")
    echo "PASS: 生成 $LINES_RAW 条原始数据"
    PASS=$((PASS + 1))
else
    echo "FAIL: generate_data.py 报错"
    FAIL=$((FAIL + 1))
fi

# Step C2: PPL 打分
echo ""
echo "===== [Test 4/4] Exp C Step 2+3: score_perplexity.py + filter_perplexity.py ====="
OUT_C_SCORED="$TEST_DIR/expC_scored.jsonl"
OUT_C_FINAL="$TEST_DIR/expC_filtered.jsonl"
if python scripts/score_perplexity.py \
    --input_file "$OUT_C_RAW" \
    --output_file "$OUT_C_SCORED" \
    --model_path "$BASE_MODEL" \
    --prompt_tokens 512 \
    --batch_size 8; then
    LINES_SCORED=$(wc -l < "$OUT_C_SCORED")
    echo "PPL 打分完成: $LINES_SCORED 条"

    # Step C3: PPL 过滤
    if python scripts/filter_perplexity.py \
        --input_file "$OUT_C_SCORED" \
        --output_file "$OUT_C_FINAL" \
        --top_percent 25; then
        LINES_FINAL=$(wc -l < "$OUT_C_FINAL")
        echo "PASS: PPL 过滤后 $LINES_FINAL 条 (25% of $LINES_SCORED)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: filter_perplexity.py 报错"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: score_perplexity.py 报错"
    FAIL=$((FAIL + 1))
fi

# ========== 汇总 ==========
echo ""
echo "====================================="
echo "Smoke test 结果: $PASS passed, $FAIL failed (共 $((PASS + FAIL)))"
echo "====================================="

# 展示输出文件
echo ""
echo "输出文件:"
ls -lh "$TEST_DIR"/*.jsonl 2>/dev/null || true

# 抽样检查内容格式
echo ""
echo "--- Exp A 第一条样本 (前 200 字符) ---"
head -1 "$OUT_A" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys())); print(d['content'][:200])" 2>/dev/null || echo "(无)"

echo ""
echo "--- Exp C scored 第一条样本 (检查 ppl 字段) ---"
head -1 "$OUT_C_SCORED" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(f'keys={list(d.keys())}, ppl={d.get(\"ppl\", \"MISSING\")}')" 2>/dev/null || echo "(无)"

echo ""
echo "--- Exp C filtered 第一条样本 (应无 ppl 字段) ---"
head -1 "$OUT_C_FINAL" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(f'keys={list(d.keys())}, has_ppl={\"ppl\" in d}')" 2>/dev/null || echo "(无)"

[ $FAIL -eq 0 ] && exit 0 || exit 1
