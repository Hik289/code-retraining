#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash src/submit_probe_train_batch_size.sh {santacoder|starcoder2|qwen25|codellama|all}"
    exit 1
fi

TARGET="$1"

submit_one() {
    local model="$1"
    sbatch --job-name="v2_probe_train_${model}" src/probe_train_batch_size.sh "$model"
}

case "$TARGET" in
    santacoder|starcoder2|qwen25|codellama)
        submit_one "$TARGET"
        ;;
    all)
        submit_one santacoder
        submit_one starcoder2
        submit_one qwen25
        submit_one codellama
        ;;
    *)
        echo "Unknown target: $TARGET"
        exit 1
        ;;
esac
