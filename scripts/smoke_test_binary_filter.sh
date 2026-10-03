#!/bin/bash

#SBATCH --job-name=smoke_binary
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --output=selfplay_results/logs/smoke_binary_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export WANDB_MODE=disabled
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

BASE_MODEL="bigcode/santacoder"
EXISTING_DATA="selfplay_results/ppl_filter/generated_data/round1_raw.jsonl"
TEST_DIR="selfplay_results/smoke_binary"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR" selfplay_results/logs

PASS=0
FAIL=0

echo ""
echo "===== [Test 1/3] Token ID check ====="
if python - <<'EOF'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bigcode/santacoder", trust_remote_code=True)
template_ids = tok.encode("\n# quality: ", add_special_tokens=False)
good_ids = tok.encode(" good", add_special_tokens=False)
bad_ids  = tok.encode(" bad",  add_special_tokens=False)
print(f'Template "\\n# quality: " => {len(template_ids)} tokens: {template_ids}')
print(f'" good" => {good_ids} ({"single token OK" if len(good_ids)==1 else "MULTI-TOKEN WARNING"})')
print(f'" bad"  => {bad_ids}  ({"single token OK" if len(bad_ids)==1 else "MULTI-TOKEN WARNING"})')
assert len(good_ids) >= 1 and len(bad_ids) >= 1
EOF
then
    echo "PASS: token ID check"
    PASS=$((PASS + 1))
else
    echo "FAIL: token ID check"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "===== [Test 2/3] score_binary_classifier.py (20 samples) ====="
head -n 20 "$EXISTING_DATA" > "$TEST_DIR/mini.jsonl"
SCORED="$TEST_DIR/mini_scored.jsonl"

if python scripts/score_binary_classifier.py \
    --input_file "$TEST_DIR/mini.jsonl" \
    --output_file "$SCORED" \
    --model_path "$BASE_MODEL" \
    --batch_size 4; then
    python - <<'EOF'
import json, math
samples = [json.loads(l) for l in open("selfplay_results/smoke_binary/mini_scored.jsonl")]
scores = [s["score"] for s in samples]
assert all(math.isfinite(s) for s in scores), "some scores are nan/inf"
assert max(scores) != min(scores), "all scores are identical"
assert all("content" in s for s in samples), "missing content field"
print(f"scores: min={min(scores):.4f}  max={max(scores):.4f}  mean={sum(scores)/len(scores):.4f}")
print(f"good (>0): {sum(s>0 for s in scores)} / {len(scores)}")
EOF
    echo "PASS: score_binary_classifier.py"
    PASS=$((PASS + 1))
else
    echo "FAIL: score_binary_classifier.py"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "===== [Test 3/3] filter_binary_classifier.py (score > 0) ====="
FILTERED="$TEST_DIR/mini_filtered.jsonl"

if python scripts/filter_binary_classifier.py \
    --input_file "$SCORED" \
    --output_file "$FILTERED" \
    --threshold 0.0; then
    python - <<'EOF'
import json
lines = [l.strip() for l in open("selfplay_results/smoke_binary/mini_filtered.jsonl") if l.strip()]
assert len(lines) > 0, "filter output is empty"
for l in lines:
    s = json.loads(l)
    assert "score" not in s, f'"score" field not removed'
    assert "content" in s, f'"content" field missing'
print(f"filtered: {len(lines)} samples, no 'score' field, 'content' present")
EOF
    echo "PASS: filter_binary_classifier.py"
    PASS=$((PASS + 1))
else
    echo "FAIL: filter_binary_classifier.py"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "====================================="
echo "Smoke test: $PASS passed, $FAIL failed (total $((PASS + FAIL)))"
echo "====================================="

echo ""
echo "--- scored sample (first line, check 'score' field) ---"
head -1 "$SCORED" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(f'keys={list(d.keys())}, score={d.get(\"score\", \"MISSING\"):.4f}')" 2>/dev/null || echo "(none)"

echo ""
echo "--- filtered sample (first line, 'score' should be absent) ---"
head -1 "$FILTERED" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(f'keys={list(d.keys())}, has_score={\"score\" in d}')" 2>/dev/null || echo "(none)"

[ $FAIL -eq 0 ] && exit 0 || exit 1
