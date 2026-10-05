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



def check_compile(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def check_repetition(code: str, threshold: float = 0.5) -> bool:
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rate = repeated / len(lines)
    return rate <= threshold


def check_length(code: str, prompt_tokens: int, tokenizer,
                 min_completion_tokens: int = 50) -> bool:
    tokens = tokenizer(code, truncation=False)["input_ids"]
    completion_len = len(tokens) - prompt_tokens
    return completion_len >= min_completion_tokens


def apply_filter(text, filter_mode, prompt_tokens, tokenizer,
                 repetition_threshold=0.3, min_completion_tokens=50):
    if filter_mode is None:
        return True, None

    if filter_mode in ("compile", "compile+quality") and not check_compile(text):
        return False, "compile"

    if filter_mode in ("quality", "compile+quality"):
        if not check_repetition(text, threshold=repetition_threshold):
            return False, "repetition"
        if not check_length(text, prompt_tokens, tokenizer,
                            min_completion_tokens=min_completion_tokens):
            return False, "length"

    return True, None



DEFAULT_ARROW_CACHE = os.environ.get("THE_STACK_ARROW_CACHE", "")


def load_dataset_iter(args):
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



def parse_args():
    parser = argparse.ArgumentParser(description="Self-play data generation")
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML (e.g. configs/santacoder.yaml)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model ID or local checkpoint path")
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default="data/python")
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="Local Arrow cache directory (overrides default)")

    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_file_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--shuffle_buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--filter_mode", type=str, default=None,
                        choices=["compile", "quality", "compile+quality"],
                        help="Inline filter mode (omit for no filtering)")
    parser.add_argument("--repetition_threshold", type=float, default=0.3)
    parser.add_argument("--min_completion_tokens", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_model_config(args.config)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    data_iter = load_dataset_iter(args)

    total_generated = 0
    passed = 0
    reject_counts = Counter()

    mode_str = args.filter_mode or "no_filter"
    with open(args.output_file, "w") as fout:
        pbar = tqdm(total=args.num_samples, desc=f"Generating ({mode_str})")

        while passed < args.num_samples:
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

    rate = passed / total_generated * 100 if total_generated > 0 else 0
    print(f"\n===== Generation stats ({mode_str}) =====")
    print(f"Model: {cfg['short_name']} ({args.model_path})")
    print(f"Total generated: {total_generated}")
    print(f"Passed: {passed} ({rate:.1f}%)")
    for reason, cnt in sorted(reject_counts.items()):
        print(f"Rejected [{reason}]: {cnt}")
    print(f"Output: {args.output_file}")

    return {
        "total_generated": total_generated,
        "passed": passed,
        "filter_pass_rate": passed / total_generated if total_generated > 0 else 0,
    }


if __name__ == "__main__":
    main()
