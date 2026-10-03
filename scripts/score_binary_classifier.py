import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE = "\n# quality: "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_content_tokens", type=int, default=2000,
                        help="Keep last N tokens of content; total seq length = N + len(template)")
    return parser.parse_args()


def score_batch(texts, model, tokenizer, template_ids, good_id, bad_id,
                max_content_tokens, device):
    seqs = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[-max_content_tokens:]
        ids = ids + template_ids
        seqs.append(ids)

    max_len = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id

    input_ids_list = []
    attention_masks = []
    for s in seqs:
        pad_len = max_len - len(s)
        input_ids_list.append([pad_id] * pad_len + s)
        attention_masks.append([0] * pad_len + [1] * len(s))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    last_logits = outputs.logits[:, -1, :]
    scores = (last_logits[:, good_id] - last_logits[:, bad_id]).tolist()
    return scores


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    template_ids = tokenizer.encode(TEMPLATE, add_special_tokens=False)
    print(f"Template: {repr(TEMPLATE)} => {len(template_ids)} tokens: {template_ids}")

    good_ids = tokenizer.encode(" good", add_special_tokens=False)
    bad_ids  = tokenizer.encode(" bad",  add_special_tokens=False)
    good_id  = good_ids[0]
    bad_id   = bad_ids[0]
    print(f'" good" => token IDs {good_ids}, using first: {good_id}')
    print(f'" bad"  => token IDs {bad_ids}, using first:  {bad_id}')
    if len(good_ids) > 1 or len(bad_ids) > 1:
        print("WARNING: multi-token target word; only using first subword logit")

    all_scores = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        scores = score_batch(
            texts, model, tokenizer, template_ids,
            good_id, bad_id, args.max_content_tokens, device,
        )
        all_scores.extend(scores)

    with open(args.output_file, "w") as f:
        for sample, score in zip(samples, all_scores):
            sample["score"] = score
            f.write(json.dumps(sample) + "\n")

    finite_scores = [s for s in all_scores if math.isfinite(s)]
    if finite_scores:
        arr = np.array(finite_scores)
        print("\n===== Score Statistics =====")
        print(f"Total: {len(all_scores)} (finite: {len(finite_scores)}, nan/inf: {len(all_scores) - len(finite_scores)})")
        print(f"min:    {arr.min():.4f}")
        print(f"p25:    {np.percentile(arr, 25):.4f}")
        print(f"median: {np.median(arr):.4f}")
        print(f"p75:    {np.percentile(arr, 75):.4f}")
        print(f"max:    {arr.max():.4f}")
        print(f"mean:   {arr.mean():.4f}")
        print(f"good (score>0): {(arr > 0).sum()} ({100*(arr > 0).mean():.1f}%)")
    print(f"Written: {args.output_file}")


if __name__ == "__main__":
    main()
