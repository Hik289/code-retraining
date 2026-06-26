"""train.py — SantaCoder 微调训练脚本

支持三种数据源：
  1. HuggingFace 数据集（The Stack 等），通过 --dataset_name 指定
  2. 本地 JSONL 文件（self-play 生成数据），通过 --local_data_path 指定
  3. 本地 Arrow 缓存（绕过 gated dataset 认证），通过 --local_arrow_path 指定

要求 transformers==4.35.2 以保证与 SantaCoder 自定义模型代码的兼容性。
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import fim


# ============================================================
# 参数解析
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()

    # 模型
    parser.add_argument("--model_path", type=str, default="bigcode/santacoder")

    # 数据集（三选一）
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集子目录，如 data/python")
    parser.add_argument("--data_column", type=str, default="content")
    parser.add_argument("--local_data_path", type=str, default=None,
                        help="本地 JSONL 文件路径。若设置，覆盖 --dataset_name")
    parser.add_argument("--local_arrow_path", type=str, default=None,
                        help="本地 Arrow 缓存目录。若设置，覆盖 --dataset_name")

    # 训练
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # FIM
    parser.add_argument("--fim_rate", type=float, default=0.5)
    parser.add_argument("--fim_spm_rate", type=float, default=0.5)

    # 数据加载
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)

    # 输出与日志
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=100)

    # 梯度检查点（默认开启）
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    return parser.parse_args()


# ============================================================
# 工具函数
# ============================================================
def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """采样估算字符/token 比率。"""
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example[data_column]
        total_characters += len(text)
        total_tokens += len(tokenizer(text).tokens())
    return total_characters / total_tokens


# ============================================================
# ConstantLengthDataset — 定长序列数据集
# ============================================================
class ConstantLengthDataset(IterableDataset):
    """将变长文本流打包成定长 token 序列。

    流程：填充字符 buffer → 批量 tokenize → 可选 FIM 变换 →
         用 EOS 拼接 → 切成 seq_length 定长序列 → 打乱。
    labels = input_ids（CLM next-token prediction）。
    """

    def __init__(self, tokenizer, dataset, infinite=False, seq_length=2048,
                 num_of_sequences=1024, chars_per_token=3.6,
                 content_field="content", fim_rate=0.5, fim_spm_rate=0.5,
                 seed=0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        self.concat_token_id = tokenizer.eos_token_id
        if self.concat_token_id is None:
            self.concat_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences

        # FIM token IDs
        self.suffix_tok_id = None
        if fim_rate > 0:
            (self.suffix_tok_id, self.prefix_tok_id,
             self.middle_tok_id, self.pad_tok_id) = fim.get_fim_token_ids(tokenizer)
            if self.suffix_tok_id is None:
                print("WARNING: tokenizer 不含 FIM token，自动禁用 FIM")
                self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            buffer, buffer_len = [], 0
            while buffer_len < self.max_buffer_size:
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]

            all_token_ids = []
            np_rng = np.random.RandomState(seed=self.seed)

            for tokenized_input in tokenized_inputs:
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        np.array(tokenized_input), np_rng,
                        self.suffix_tok_id, self.prefix_tok_id,
                        self.middle_tok_id, self.pad_tok_id,
                        self.fim_rate, self.fim_spm_rate,
                    )
                    tokenized_input = tokenized_input.tolist()

                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)

            random.shuffle(examples)

            for example in examples:
                yield {
                    "input_ids": torch.tensor(example, dtype=torch.long),
                    "labels": torch.tensor(example, dtype=torch.long),
                }


# ============================================================
# 数据集创建
# ============================================================
def create_datasets(tokenizer, args):
    if args.local_data_path:
        # 本地 JSONL 文件
        dataset = load_dataset("json",
                               data_files={"train": args.local_data_path},
                               split="train")
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    elif args.local_arrow_path:
        # 本地 Arrow 缓存（绕过 gated dataset 认证）
        data_files = sorted([
            f for f in glob.glob(os.path.join(args.local_arrow_path, "*.arrow"))
            if not os.path.basename(f).startswith("cache-")
        ])
        print(f"从本地 Arrow 缓存加载，共 {len(data_files)} 个 shard")
        dataset = load_dataset("arrow", data_files=data_files, split="train",
                               streaming=True)
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    elif args.streaming:
        dataset = load_dataset(args.dataset_name, data_dir=args.data_dir,
                               split="train", streaming=True)
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
    else:
        dataset = load_dataset(args.dataset_name, data_dir=args.data_dir,
                               split="train", num_proc=args.num_workers)
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"字符/token 比率: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length,
        content_field=args.data_column, chars_per_token=chars_per_token,
        fim_rate=args.fim_rate, fim_spm_rate=args.fim_spm_rate, seed=args.seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length,
        content_field=args.data_column, chars_per_token=chars_per_token,
        fim_rate=0, seed=args.seed,
    )
    return train_dataset, valid_dataset


# ============================================================
# 训练
# ============================================================
def run_training(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_cache=not args.gradient_checkpointing,
    )

    train_dataset, valid_dataset = create_datasets(tokenizer, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        dataloader_drop_last=True,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=valid_dataset,
    )

    trainer.train()
    final_ckpt = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    # 训练时 use_cache=False，但推理需要 KV cache，修正 generation_config
    from transformers import GenerationConfig
    gen_config = GenerationConfig.from_pretrained(final_ckpt)
    gen_config.use_cache = True
    gen_config.save_pretrained(final_ckpt)
    print(f"训练完成，最终 checkpoint 保存至 {final_ckpt}")


if __name__ == "__main__":
    args = get_args()
    run_training(args)
