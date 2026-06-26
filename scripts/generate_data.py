"""scripts/generate_data.py — Self-play 数据生成脚本

从 The Stack Python 流式加载，取每个文件前 1024 tokens 作 prompt，
用模型生成后续 1024 tokens，保存为训练兼容的 JSONL。

用法：
    python scripts/generate_data.py \
        --model_path bigcode/santacoder \
        --output_file selfplay_results/generated_data/round1.jsonl \
        --num_samples 5000 --seed 1

要求 transformers==4.35.2。
"""
import argparse
import glob
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    # 数据源（二选一）
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default="data/python")
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="本地 Arrow 缓存目录（HF Hub 不可达时使用）")

    # 生成参数
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_file_tokens", type=int, default=512,
                        help="跳过 token 数少于此值的文件")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    # 采样
    parser.add_argument("--shuffle_buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1,
                        help="每轮传不同 seed，实现近似随机采样")
    return parser.parse_args()


def load_dataset_iter(args):
    """加载数据集，返回流式迭代器。"""
    if args.local_dataset_path:
        # 本地 Arrow 缓存：过滤掉 cache-*.arrow 索引文件（schema 不兼容）
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token   # SantaCoder 无 pad_token
    tokenizer.padding_side = "left"             # causal LM 生成要求左侧 padding

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

    # ---- 批量推理 ----
    count = 0
    batch_prompts = []

    with open(args.output_file, "w") as fout:
        pbar = tqdm(total=args.num_samples, desc="Generating")

        while count < args.num_samples:
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
                fout.write(json.dumps({"content": full_text}) + "\n")
                count += 1
                pbar.update(1)
                if count >= args.num_samples:
                    break

            batch_prompts = []

        pbar.close()

    print(f"完成：{count} 条样本写入 {args.output_file}")


if __name__ == "__main__":
    main()
