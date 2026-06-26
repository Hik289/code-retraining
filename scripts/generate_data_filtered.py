"""scripts/generate_data_filtered.py — Self-play 数据生成 + 在线过滤

与 generate_data.py 相同的生成逻辑，但每批生成后立即做过滤，
只保留通过过滤的样本。循环生成直到过滤后累积达到 num_samples 条。

支持的过滤模式：
  --filter_mode compile        Exp A: compile() 语法检查
  --filter_mode compile+quality  Exp B: compile + 重复度 + 长度过滤

用法：
    python scripts/generate_data_filtered.py \
        --model_path bigcode/santacoder \
        --output_file selfplay_results/generated_data/round1_filtered.jsonl \
        --num_samples 5000 --seed 1 \
        --filter_mode compile
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


# ===================== Filters =====================

def check_compile(code: str) -> bool:
    """Return True if code passes compile() syntax check."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def check_repetition(code: str, threshold: float = 0.5) -> bool:
    """Return True if line-level repetition rate is below threshold.

    Line-level repetition = fraction of non-empty lines that are duplicates.
    Code naturally has repeated char n-grams (indentation, keywords), so
    line-level is a better signal for degenerate repetition.
    """
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rate = repeated / len(lines)
    return rate <= threshold


def check_length(code: str, prompt_tokens: int, tokenizer, min_completion_tokens: int = 50) -> bool:
    """Return True if completion part has at least min_completion_tokens tokens."""
    tokens = tokenizer(code, truncation=False)["input_ids"]
    completion_len = len(tokens) - prompt_tokens
    return completion_len >= min_completion_tokens


# ===================== Args =====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    # 数据源
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default="data/python")
    parser.add_argument("--local_dataset_path", type=str, default=None)

    # 生成参数
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="过滤后需要的样本数")
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--min_file_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    # 采样
    parser.add_argument("--shuffle_buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)

    # 过滤
    parser.add_argument("--filter_mode", type=str, default="compile",
                        choices=["compile", "compile+quality"],
                        help="compile: 仅语法检查; compile+quality: 语法+重复度+长度")
    parser.add_argument("--repetition_threshold", type=float, default=0.5,
                        help="行级重复率上限 (compile+quality 模式)")
    parser.add_argument("--min_completion_tokens", type=int, default=50,
                        help="completion 部分最少 token 数 (compile+quality 模式)")

    return parser.parse_args()


def load_dataset_iter(args):
    """加载数据集，返回流式迭代器。"""
    if args.local_dataset_path:
        data_files = sorted([
            f for f in glob.glob(os.path.join(args.local_dataset_path, "*.arrow"))
            if not os.path.basename(f).startswith("cache-")
        ])
        if not data_files:
            raise FileNotFoundError(
                f"在 {args.local_dataset_path} 中未找到非 cache 的 .arrow 文件"
            )
        dataset = load_dataset("arrow", data_files=data_files,
                               split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset_name, data_dir=args.data_dir,
                               split="train", streaming=True)

    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    return iter(dataset)


def apply_filter(text, args, tokenizer):
    """Apply filter chain. Returns (pass: bool, reject_reason: str or None)."""
    if not check_compile(text):
        return False, "compile"

    if args.filter_mode == "compile+quality":
        if not check_repetition(text, threshold=args.repetition_threshold):
            return False, "repetition"
        if not check_length(text, args.prompt_tokens, tokenizer,
                            min_completion_tokens=args.min_completion_tokens):
            return False, "length"

    return True, None


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    # ---- 数据 ----
    data_iter = load_dataset_iter(args)

    # ---- 统计 ----
    total_generated = 0
    passed = 0
    reject_counts = Counter()

    # ---- 循环生成+过滤 ----
    with open(args.output_file, "w") as fout:
        pbar = tqdm(total=args.num_samples, desc=f"Generating ({args.filter_mode})")

        while passed < args.num_samples:
            # 收集一批 prompt
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
                print("WARNING: 数据集耗尽，提前停止")
                break

            # 生成
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

            # 逐条过滤
            for seq in outputs:
                full_text = tokenizer.decode(seq, skip_special_tokens=True)
                total_generated += 1

                ok, reason = apply_filter(full_text, args, tokenizer)
                if ok:
                    fout.write(json.dumps({"content": full_text}) + "\n")
                    passed += 1
                    pbar.update(1)
                    if passed >= args.num_samples:
                        break
                else:
                    reject_counts[reason] += 1

            batch_prompts = []

        pbar.close()

    # ---- 打印统计 ----
    rate = passed / total_generated * 100 if total_generated > 0 else 0
    print(f"\n===== 过滤统计 ({args.filter_mode}) =====")
    print(f"总生成: {total_generated}")
    print(f"通过:   {passed} ({rate:.1f}%)")
    for reason, cnt in sorted(reject_counts.items()):
        print(f"拒绝 [{reason}]: {cnt}")
    print(f"写入: {args.output_file}")


if __name__ == "__main__":
    main()