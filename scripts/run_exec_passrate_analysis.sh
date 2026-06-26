#!/bin/bash
# Analyze exec() pass rate across different rounds to assess feasibility of exec filter
#
#SBATCH --job-name=exec_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=h200
#SBATCH --time=02:00:00
#SBATCH --output=selfplay_results/logs/exec_analysis_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

MAX_SAMPLES=500
TIMEOUT=5

echo "====== exec() Pass Rate Analysis ======"
echo "max_samples=$MAX_SAMPLES  timeout=${TIMEOUT}s"
echo ""

run_analysis() {
    local label="$1"
    local file="$2"
    if [ -f "$file" ]; then
        echo "=========================================="
        echo "[$label]  $file"
        echo "=========================================="
        python scripts/analyze_exec_passrate.py \
            --input_file "$file" \
            --timeout "$TIMEOUT" \
            --max_samples "$MAX_SAMPLES"
        echo ""
    else
        echo "SKIP [$label]: $file not found"
        echo ""
    fi
}

# No-filter self-play (raw collapse data)
run_analysis "no-filter R1" "selfplay_results/generated_data/round1.jsonl"
run_analysis "no-filter R5" "selfplay_results/generated_data/round5.jsonl"

# Compile-filter (already syntax-clean)
run_analysis "compile-filter R1" "selfplay_results/compile_filter/generated_data/round1.jsonl"
run_analysis "compile-filter R10" "selfplay_results/compile_filter/generated_data/round10.jsonl"

# PPL-filter
run_analysis "ppl-filter R1" "selfplay_results/ppl_filter/generated_data/round1.jsonl"
run_analysis "ppl-filter R8" "selfplay_results/ppl_filter/generated_data/round8.jsonl"

echo "====== Analysis Complete ======"
