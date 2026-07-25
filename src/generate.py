"""Multi-model self-play data generation.

Streams Python files from The Stack, uses the first ``prompt_tokens`` tokens
as context, generates ``max_new_tokens`` continuation tokens, and writes JSONL.

Two modes are supported:
  1. Without ``--filter_mode``, generate ``num_samples`` records once.
  2. With ``compile`` or ``compile+quality``, continue until ``num_samples``
     records pass the selected filter.

Perplexity and binary filtering are handled separately by ``src/filters.py``.

Usage:
    # No inline filter
    python src/generate.py --config configs/santacoder.yaml \
        --model_path bigcode/santacoder \
        --output_file results/santacoder/no_filter/generated_data/round1.jsonl \
        --num_samples 5000 --seed 1

    # Compile filter
    python src/generate.py --config configs/santacoder.yaml \
        --model_path bigcode/santacoder \
        --output_file results/santacoder/compile_filter/generated_data/round1.jsonl \
        --num_samples 5000 --seed 1 --filter_mode compile
"""
import argparse
import glob
import json
import os
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_model_config


# ===================== Inline Filters =====================

def check_compile(code: str) -> bool:
    """Return True if code passes compile() syntax check."""
    try:
        compile(code, "<string>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def check_repetition(code: str, threshold: float = 0.5) -> bool:
    """Return True if line-level repetition rate is below threshold."""
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rate = repeated / len(lines)
    return rate <= threshold


def check_length(code: str, prompt_tokens: int, tokenizer,
                 min_completion_tokens: int = 50) -> bool:
    """Return True if completion part has at least min_completion_tokens tokens."""
    tokens = tokenizer(code, truncation=False)["input_ids"]
    completion_len = len(tokens) - prompt_tokens
    return completion_len >= min_completion_tokens


def apply_filter(text, filter_mode, prompt_tokens, tokenizer,
                 repetition_threshold=0.5, min_completion_tokens=50):
    """Apply inline filter chain.

    Returns:
        (passed: bool, reject_reason: str or None)
    """
    if filter_mode is None:
        return True, None

    if not check_compile(text):
        return False, "compile"

    if filter_mode == "compile+quality":
        if not check_repetition(text, threshold=repetition_threshold):
            return False, "repetition"
        if not check_length(text, prompt_tokens, tokenizer,
                            min_completion_tokens=min_completion_tokens):
            return False, "length"

    return True, None


# ===================== Data Loading =====================

# Default Arrow cache path for The Stack
# Path to a locally cached The Stack (dedup, Python) Arrow dataset directory.
# Set the THE_STACK_ARROW_CACHE environment variable to point at it, or pass
# --local_dataset_path on the command line. Falls back to streaming from the Hub.
DEFAULT_ARROW_CACHE = os.environ.get("THE_STACK_ARROW_CACHE", "")


def load_dataset_iter(args):
    """Load The Stack and return a shuffled streaming iterator."""
    if args.local_dataset_path:
        arrow_dir = args.local_dataset_path
    elif os.path.isdir(DEFAULT_ARROW_CACHE):
        arrow_dir = DEFAULT_ARROW_CACHE
    else:
        arrow_dir = None

    if arrow_dir:
        data_files = sorted([
            f for f in glob.glob(os.path.join(arrow_dir, "*.arrow"))
            if not os.path.basename(f).startswith("cache-")
        ])
        if not data_files:
            raise FileNotFoundError(f"No .arrow files found in {arrow_dir}")
        dataset = load_dataset("arrow", data_files=data_files,
                               split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset_name, data_dir=args.data_dir,
                               split="train", streaming=True)

    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    return iter(dataset)


# ===================== Main =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Self-play data generation")
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML (e.g. configs/santacoder.yaml)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model ID or local checkpoint path")
    parser.add_argument("--output_file", type=str, required=True)

    # Data source
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default="data/python")
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="Local Arrow cache directory (overrides default)")

    # Generation params
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_file_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Sampling
    parser.add_argument("--shuffle_buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)

    # Filter (optional — only for compile / compile+quality)
    parser.add_argument("--filter_mode", type=str, default=None,
                        choices=["compile", "compile+quality"],
                        help="Inline filter mode (omit for no filtering)")
    parser.add_argument("--repetition_threshold", type=float, default=0.5)
    parser.add_argument("--min_completion_tokens", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_model_config(args.config)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # causal LM generation requires left padding

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    # ---- Data ----
    data_iter = load_dataset_iter(args)

    # ---- Generate ----
    total_generated = 0
    passed = 0
    reject_counts = Counter()

    mode_str = args.filter_mode or "no_filter"
    with open(args.output_file, "w") as fout:
        pbar = tqdm(total=args.num_samples, desc=f"Generating ({mode_str})")

        while passed < args.num_samples:
            # Collect batch of prompts
            batch_prompts = []
            while len(batch_prompts) < args.batch_size:
                try:
                    example = next(data_iter)
                except StopIteration:
                    break
                text = example["content"]
                tokens = tokenizer(text, truncation=False)["input_ids"]
                if len(tokens) < args.min_file_tokens:
                    continue
                prompt_ids = tokens[: args.prompt_tokens]
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                batch_prompts.append(prompt_text)

            if not batch_prompts:
                print("WARNING: dataset exhausted, stopping early")
                break

            # Generate
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.prompt_tokens,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_cache=True,
                )

            # Write (with optional filter)
            for seq in outputs:
                full_text = tokenizer.decode(seq, skip_special_tokens=True)
                total_generated += 1

                ok, reason = apply_filter(
                    full_text, args.filter_mode, args.prompt_tokens, tokenizer,
                    args.repetition_threshold, args.min_completion_tokens,
                )
                if ok:
                    fout.write(json.dumps({"content": full_text}) + "\n")
                    passed += 1
                    pbar.update(1)
                    if passed >= args.num_samples:
                        break
                else:
                    reject_counts[reason] += 1

        pbar.close()

    # ---- Stats ----
    rate = passed / total_generated * 100 if total_generated > 0 else 0
    print(f"\n===== Generation stats ({mode_str}) =====")
    print(f"Model: {cfg['short_name']} ({args.model_path})")
    print(f"Total generated: {total_generated}")
    print(f"Passed: {passed} ({rate:.1f}%)")
    for reason, cnt in sorted(reject_counts.items()):
        print(f"Rejected [{reason}]: {cnt}")
    print(f"Output: {args.output_file}")

    # Return stats for run_experiment.sh to capture
    return {
        "total_generated": total_generated,
        "passed": passed,
        "filter_pass_rate": passed / total_generated if total_generated > 0 else 0,
    }


if __name__ == "__main__":
    main()
