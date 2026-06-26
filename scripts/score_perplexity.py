"""scripts/score_perplexity.py — 计算生成数据的 completion 部分 perplexity

对每条样本，只计算 completion 部分（prompt 之后）的 PPL。
prompt 部分的 label 设为 -100（忽略）。

用法：
    python scripts/score_perplexity.py \
        --input_file selfplay_results/generated_data/round1.jsonl \
        --output_file selfplay_results/generated_data/round1_scored.jsonl \
        --model_path bigcode/santacoder \
        --prompt_tokens 1024
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_tokens", type=int, default=1024,
                        help="prompt 部分的 token 数（这部分 loss 不计入 PPL）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="截断到最大长度")
    return parser.parse_args()


def compute_ppl_batch(texts, model, tokenizer, prompt_tokens, max_length, device):
    """Compute per-sample PPL for a batch, only on completion tokens."""
    encodings = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(device)

    input_ids = encodings["input_ids"]          # (B, L)
    attention_mask = encodings["attention_mask"] # (B, L)

    # Labels: copy input_ids, mask prompt and padding
    labels = input_ids.clone()
    # Mask padding
    labels[attention_mask == 0] = -100
    # Mask prompt tokens (first prompt_tokens non-padding tokens per sample)
    # Since padding_side="left", real tokens start from the right
    for i in range(labels.size(0)):
        # Find where real tokens start
        real_start = (attention_mask[i] == 0).sum().item()
        # Mask prompt portion
        mask_end = min(real_start + prompt_tokens, labels.size(1))
        labels[i, :mask_end] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)

    # outputs.loss is averaged over all non-ignored tokens across the batch
    # We need per-sample loss, so compute manually
    logits = outputs.logits  # (B, L, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # (B, L-1)
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    # Per-sample mean loss (only over non-ignored tokens)
    mask = shift_labels != -100  # (B, L-1)
    ppls = []
    for i in range(input_ids.size(0)):
        sample_mask = mask[i]
        if sample_mask.sum() == 0:
            ppls.append(float("inf"))
        else:
            mean_loss = per_token_loss[i][sample_mask].mean().item()
            ppls.append(math.exp(mean_loss))

    return ppls


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load data
    samples = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    # Score in batches
    all_ppls = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring PPL"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        ppls = compute_ppl_batch(texts, model, tokenizer,
                                 args.prompt_tokens, args.max_length, device)
        all_ppls.extend(ppls)

    # Write output
    with open(args.output_file, "w") as f:
        for sample, ppl in zip(samples, all_ppls):
            sample["ppl"] = ppl
            f.write(json.dumps(sample) + "\n")

    # Statistics
    finite_ppls = [p for p in all_ppls if math.isfinite(p)]
    if finite_ppls:
        arr = np.array(finite_ppls)
        print(f"\n===== PPL 统计 =====")
        print(f"样本数: {len(all_ppls)} (有效: {len(finite_ppls)}, inf: {len(all_ppls) - len(finite_ppls)})")
        print(f"min:    {arr.min():.2f}")
        print(f"p25:    {np.percentile(arr, 25):.2f}")
        print(f"median: {np.median(arr):.2f}")
        print(f"p75:    {np.percentile(arr, 75):.2f}")
        print(f"max:    {arr.max():.2f}")
        print(f"mean:   {arr.mean():.2f}")
    print(f"写入: {args.output_file}")


if __name__ == "__main__":
    main()
