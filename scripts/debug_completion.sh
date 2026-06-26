#!/bin/bash
# 调试：检查 SantaCoder 在 HumanEval 上的原始生成输出
# 用法：sbatch scripts/debug_completion.sh

#SBATCH --job-name=debug_completion
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --output=selfplay_results/logs/debug_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/venv/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

tasks = get_human_eval_plus()

tokenizer = AutoTokenizer.from_pretrained('bigcode/santacoder', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('bigcode/santacoder', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

STOP_SEQUENCES = ['\nclass', '\ndef', '\n#', '\n@', '\nprint', '\nif', '\n\`\`\`', '<file_sep>', '<|endoftext|>']

def truncate_at_stop(text, stop_sequences):
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]

# 测试前 5 个 task
for i, (task_id, task) in enumerate(tasks.items()):
    if i >= 5:
        break
    prompt = task['prompt']
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    generated_ids = output[0, input_ids.size(-1):]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    truncated = truncate_at_stop(raw, STOP_SEQUENCES)

    print(f'===== {task_id} =====')
    print(f'Prompt ends with: {repr(prompt[-50:])}')
    print(f'Raw completion ({len(raw)} chars): {repr(raw[:300])}')
    print(f'Truncated ({len(truncated)} chars): {repr(truncated[:300])}')
    print()
"
