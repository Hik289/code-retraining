"""filters.py — 离线过滤：compile / quality / PPL 打分 / Binary 打分 / top-K 过滤

所有过滤功能通过子命令调用：

    # compile 检查（纯 CPU）
    python src/filters.py compile \
        --input_file round1_raw.jsonl --output_file round1.jsonl

    # compile + quality（纯 CPU）
    python src/filters.py quality \
        --input_file round1_raw.jsonl --output_file round1.jsonl

    # PPL 打分（需 GPU）
    python src/filters.py score-ppl \
        --input_file round1_raw.jsonl --output_file round1_scored.jsonl \
        --model_path bigcode/santacoder --config configs/santacoder.yaml

    # Binary 打分（需 GPU）
    python src/filters.py score-binary \
        --input_file round1_raw.jsonl --output_file round1_scored.jsonl \
        --model_path bigcode/santacoder --config configs/santacoder.yaml

    # top-K 过滤（纯 CPU）
    python src/filters.py filter-topk \
        --input_file round1_scored.jsonl --output_file round1.jsonl \
        --score_field ppl --top_percent 25 --ascending
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np


# ===================== Shared Utilities =====================

def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ===================== CPU Filters =====================

def check_compile(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def check_repetition(code: str, threshold: float = 0.5) -> bool:
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rate = repeated / len(lines)
    return rate <= threshold


def check_length(code: str, tokenizer, prompt_tokens: int,
                 min_completion_tokens: int = 50) -> bool:
    tokens = tokenizer(code, truncation=False)["input_ids"]
    completion_len = len(tokens) - prompt_tokens
    return completion_len >= min_completion_tokens


# ===================== Subcommand: compile =====================

def cmd_compile(args):
    samples = load_jsonl(args.input_file)
    kept = [s for s in samples if check_compile(s["content"])]
    write_jsonl(kept, args.output_file)

    rate = len(kept) / len(samples) * 100 if samples else 0
    print(f"\n===== Compile Filter =====")
    print(f"Input:  {len(samples)}")
    print(f"Passed: {len(kept)} ({rate:.1f}%)")
    print(f"Output: {args.output_file}")


# ===================== Subcommand: quality =====================

def cmd_quality(args):
    from transformers import AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path or cfg["model_id"],
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    samples = load_jsonl(args.input_file)
    kept = []
    reject_counts = Counter()

    for s in samples:
        text = s["content"]
        if not check_compile(text):
            reject_counts["compile"] += 1
            continue
        if not check_repetition(text, threshold=args.repetition_threshold):
            reject_counts["repetition"] += 1
            continue
        if not check_length(text, tokenizer, args.prompt_tokens,
                            min_completion_tokens=args.min_completion_tokens):
            reject_counts["length"] += 1
            continue
        kept.append(s)

    write_jsonl(kept, args.output_file)

    rate = len(kept) / len(samples) * 100 if samples else 0
    print(f"\n===== Quality Filter (compile + repetition + length) =====")
    print(f"Input:  {len(samples)}")
    print(f"Passed: {len(kept)} ({rate:.1f}%)")
    for reason, cnt in sorted(reject_counts.items()):
        print(f"  Rejected [{reason}]: {cnt}")
    print(f"Output: {args.output_file}")


# ===================== Subcommand: score-ppl =====================

def compute_ppl_batch(texts, model, tokenizer, prompt_tokens, max_length, device):
    import torch

    encodings = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    # Mask prompt tokens; padding_side="left" so real tokens start from right
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


def cmd_score_ppl(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_ppls = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring PPL"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        ppls = compute_ppl_batch(texts, model, tokenizer,
                                 args.prompt_tokens, args.max_length, device)
        all_ppls.extend(ppls)

    for sample, ppl in zip(samples, all_ppls):
        sample["ppl"] = ppl
    write_jsonl(samples, args.output_file)

    finite_ppls = [p for p in all_ppls if math.isfinite(p)]
    if finite_ppls:
        arr = np.array(finite_ppls)
        print(f"\n===== PPL Stats =====")
        print(f"Samples: {len(all_ppls)} (finite: {len(finite_ppls)}, "
              f"inf: {len(all_ppls) - len(finite_ppls)})")
        print(f"min={arr.min():.2f}  p25={np.percentile(arr, 25):.2f}  "
              f"median={np.median(arr):.2f}  p75={np.percentile(arr, 75):.2f}  "
              f"max={arr.max():.2f}")
    print(f"Output: {args.output_file}")


# ===================== Subcommand: score-binary =====================

BINARY_TEMPLATE = "\n# quality: "


def score_binary_batch(texts, model, tokenizer, template_ids, good_id, bad_id,
                       max_content_tokens, device):
    import torch

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


def cmd_score_binary(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    # Resolve good/bad token IDs from config
    good_token = cfg.get("binary_good_token", " good")
    bad_token = cfg.get("binary_bad_token", " bad")
    good_ids = tokenizer.encode(good_token, add_special_tokens=False)
    bad_ids = tokenizer.encode(bad_token, add_special_tokens=False)
    good_id = good_ids[0]
    bad_id = bad_ids[0]
    print(f"Good token: {repr(good_token)} => IDs {good_ids}, using {good_id}")
    print(f"Bad token:  {repr(bad_token)} => IDs {bad_ids}, using {bad_id}")
    if len(good_ids) > 1 or len(bad_ids) > 1:
        print("WARNING: multi-token target; only using first subword logit")

    template_ids = tokenizer.encode(BINARY_TEMPLATE, add_special_tokens=False)
    print(f"Template: {repr(BINARY_TEMPLATE)} => {len(template_ids)} tokens")

    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_scores = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring binary"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        scores = score_binary_batch(
            texts, model, tokenizer, template_ids,
            good_id, bad_id, args.max_content_tokens, device,
        )
        all_scores.extend(scores)

    for sample, score in zip(samples, all_scores):
        sample["score"] = score
    write_jsonl(samples, args.output_file)

    finite_scores = [s for s in all_scores if math.isfinite(s)]
    if finite_scores:
        arr = np.array(finite_scores)
        print(f"\n===== Binary Score Stats =====")
        print(f"Samples: {len(all_scores)} (finite: {len(finite_scores)}, "
              f"nan/inf: {len(all_scores) - len(finite_scores)})")
        print(f"min={arr.min():.4f}  median={np.median(arr):.4f}  "
              f"max={arr.max():.4f}")
        print(f"good (score>0): {(arr > 0).sum()} ({100*(arr > 0).mean():.1f}%)")
    print(f"Output: {args.output_file}")


# ===================== Subcommand: filter-topk =====================

def cmd_filter_topk(args):
    samples = load_jsonl(args.input_file)
    field = args.score_field

    valid = [(i, s) for i, s in enumerate(samples)
             if math.isfinite(s.get(field, float("inf")))]
    invalid_count = len(samples) - len(valid)

    valid.sort(key=lambda x: x[1][field], reverse=(not args.ascending))

    keep_count = max(1, int(len(valid) * args.top_percent / 100))
    kept = valid[:keep_count]

    # Remove score field from output to keep clean training format
    out_samples = []
    for _, s in kept:
        out = {k: v for k, v in s.items() if k != field}
        out_samples.append(out)
    write_jsonl(out_samples, args.output_file)

    if valid:
        all_vals = np.array([s[field] for _, s in valid])
        kept_vals = np.array([s[field] for _, s in kept])
        order = "ascending (lower=better)" if args.ascending else "descending (higher=better)"
        print(f"\n===== Top-K Filter ({field}, {order}) =====")
        print(f"Input:   {len(samples)} (valid: {len(valid)}, invalid: {invalid_count})")
        print(f"Keeping: {keep_count} (top {args.top_percent}%)")
        print(f"All {field}:  min={all_vals.min():.4f}  median={np.median(all_vals):.4f}  "
              f"max={all_vals.max():.4f}")
        print(f"Kept {field}: min={kept_vals.min():.4f}  median={np.median(kept_vals):.4f}  "
              f"max={kept_vals.max():.4f}")
    print(f"Output: {args.output_file}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="Self-play data filtering (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- compile ----
    p = sub.add_parser("compile", help="Keep only samples that pass compile()")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)

    # ---- quality ----
    p = sub.add_parser("quality", help="compile + repetition + length check")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local path (default: config model_id)")
    p.add_argument("--prompt_tokens", type=int, default=1024)
    p.add_argument("--repetition_threshold", type=float, default=0.5)
    p.add_argument("--min_completion_tokens", type=int, default=50)

    # ---- score-ppl ----
    p = sub.add_parser("score-ppl", help="Score samples by PPL (needs GPU)")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local checkpoint (default: config model_id)")
    p.add_argument("--prompt_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)

    # ---- score-binary ----
    p = sub.add_parser("score-binary", help="Score samples by binary classifier (needs GPU)")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local checkpoint (default: config model_id)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_content_tokens", type=int, default=2000)

    # ---- filter-topk ----
    p = sub.add_parser("filter-topk", help="Keep top K% by score field")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--score_field", required=True,
                   help="JSON field to sort by (e.g. 'ppl' or 'score')")
    p.add_argument("--top_percent", type=float, default=25)
    p.add_argument("--ascending", action="store_true",
                   help="Sort ascending (lower=better, e.g. PPL). "
                        "Default: descending (higher=better, e.g. binary score)")

    args = parser.parse_args()

    dispatch = {
        "compile": cmd_compile,
        "quality": cmd_quality,
        "score-ppl": cmd_score_ppl,
        "score-binary": cmd_score_binary,
        "filter-topk": cmd_filter_topk,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
