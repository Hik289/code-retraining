#!/bin/bash
# 调试：深入检查为什么 SantaCoder 立即输出 EOS
#SBATCH --job-name=debug_comp2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --output=selfplay_results/logs/debug2_%j.out

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$PROJECT_DIR"
source venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

tasks = get_human_eval_plus()
task = tasks['HumanEval/0']
prompt = task['prompt']

tokenizer = AutoTokenizer.from_pretrained('bigcode/santacoder', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('bigcode/santacoder', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

print('=== Tokenizer info ===')
print(f'eos_token: {repr(tokenizer.eos_token)} id={tokenizer.eos_token_id}')
print(f'pad_token: {repr(tokenizer.pad_token)} id={tokenizer.pad_token_id}')
print(f'bos_token: {repr(tokenizer.bos_token)} id={getattr(tokenizer, \"bos_token_id\", None)}')
print(f'vocab_size: {tokenizer.vocab_size}')
print()

# 检查 generate 的原始 token ids
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
print(f'Prompt length: {input_ids.shape[-1]} tokens')
print(f'Last 5 prompt tokens: {input_ids[0, -5:].tolist()}')
print(f'Decoded last 5: {[tokenizer.decode([t]) for t in input_ids[0, -5:].tolist()]}')
print()

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)

gen_ids = output[0, input_ids.size(-1):]
print(f'Generated {len(gen_ids)} tokens: {gen_ids.tolist()}')
print(f'Decoded (skip_special=False): {repr(tokenizer.decode(gen_ids, skip_special_tokens=False))}')
print(f'Decoded (skip_special=True):  {repr(tokenizer.decode(gen_ids, skip_special_tokens=True))}')
print()

# 也试试 float16 而不是 bfloat16
print('=== Trying float16 ===')
del model
torch.cuda.empty_cache()
model2 = AutoModelForCausalLM.from_pretrained('bigcode/santacoder', trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
model2.eval()
with torch.no_grad():
    output2 = model2.generate(input_ids, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
gen_ids2 = output2[0, input_ids.size(-1):]
print(f'Generated {len(gen_ids2)} tokens: {gen_ids2.tolist()[:20]}')
print(f'Decoded: {repr(tokenizer.decode(gen_ids2, skip_special_tokens=False)[:300])}')
"
