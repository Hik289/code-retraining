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
    encodings = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i in range(labels.size(0)):
        real_start = (attention_mask[i] == 0).sum().item()
        mask_end = min(real_start + prompt_tokens, labels.size(1))
        labels[i, :mask_end] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    mask = shift_labels != -100
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

    samples = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.input_file}")

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

    all_ppls = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring PPL"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        ppls = compute_ppl_batch(texts, model, tokenizer,
                                 args.prompt_tokens, args.max_length, device)
        all_ppls.extend(ppls)

    with open(args.output_file, "w") as f:
        for sample, ppl in zip(samples, all_ppls):
            sample["ppl"] = ppl
            f.write(json.dumps(sample) + "\n")

    finite_ppls = [p for p in all_ppls if math.isfinite(p)]
    if finite_ppls:
        arr = np.array(finite_ppls)
        print("\n===== PPL 统计 =====")
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
