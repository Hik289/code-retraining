# Self-Play Fine-tuning 项目规格文档

> 用途：从零开始重建整个项目的唯一参考文档。假设你是一个新人，需要从互联网下载所有代码和数据，完全不依赖任何已有代码。
>
> 日期：2026-03-20

---

## 一、研究目标

观察**累积式 self-play 微调**随迭代轮次对代码生成性能的影响。

- **基础模型：** [bigcode/santacoder](https://huggingface.co/bigcode/santacoder)（~1.1B 参数，预训练语料为 Python / Java / JavaScript，预训练时使用了 FIM）
- **数据来源：** [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) 的 Python 子集
- **Self-play 做法：** 让模型对 The Stack Python 文件做前缀补全（取前 1024 tokens 作 prompt，生成后续 1024 tokens），用生成的数据微调自己，多轮迭代
- **评估：** 每轮后用 [EvalPlus](https://github.com/evalplus/evalplus)（HumanEval+, MBPP+）和 [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 评估
- **参照上限：** 直接在 The Stack 人类数据上微调 30K 步

**核心问题：** 用模型自己生成的数据迭代训练自己，性能会提升、退化还是持平？

---

## 二、实验设计

### 2.1 总体结构

```
Model_0 = bigcode/santacoder（原始，不微调）
评估 Model_0 → Round 0 分数（EvalPlus + LiveCodeBench）

Round 1：用 Model_0 生成 5000 条数据（seed=1）
         从 Model_0 继续训练 6000 步 → Model_1
         评估 Model_1（EvalPlus + LiveCodeBench）

Round 2：用 Model_1 生成 5000 条数据（seed=2）
         从 Model_1 继续训练 6000 步 → Model_2
         评估 Model_2（EvalPlus + LiveCodeBench）

...（共 5 轮，总计 30,000 步）

参照：The Stack baseline（同样 30,000 步，人类数据，一次性训练）
      评估 baseline（EvalPlus + LiveCodeBench）
```

### 2.2 关键设计约定

| 设计项 | 决策 | 理由 |
|--------|------|------|
| 总步数 | 30,000（= baseline） | 计算量对齐，比较更有说服力 |
| 轮次 | 5 轮 × 6,000 步 | 5 个评估点，趋势清晰 |
| 每轮样本数 | 5,000 条 | 重复约 19 次（6000×16/5000），可接受 |
| 训练方式 | 累积式（从上轮 checkpoint 继续） | 真正的 self-play 迭代 |
| Prompt 长度 | 文件前 1024 tokens（约 50%） | 足够长保证生成质量，足够短让生成占主导 |
| 生成长度 | 后续 1024 tokens | prompt + generated = 2048，与 seq_length 对齐 |
| FIM | 开启（`fim_rate=0.5, fim_spm_rate=0.5`，与 baseline 一致） | SantaCoder 预训练时使用了 FIM，关掉可能引入 confound；保持一致使唯一变量为数据来源 |
| 数据过滤 | 不过滤 | 观察原始 self-play 效果，过滤作为后续 ablation |
| 每轮采样 | `dataset.shuffle(buffer_size=50000, seed=round_id)` | 流式数据集不支持随机索引，shuffle buffer 近似随机 |

### 2.3 超参数对照

Self-play 与 baseline 除数据来源外完全一致，确保唯一变量为数据来源（自生成 vs 人类）。

| 参数 | Self-Play 每轮 | The Stack Baseline |
|------|--------------|-------------------|
| `seq_length` | 2048 | 2048 |
| `max_steps` | 6,000（×5 轮=30,000） | 30,000（一次性） |
| `batch_size` | 2 | 2 |
| `gradient_accumulation_steps` | 8 | 8 |
| 等效全局 batch | 16 | 16 |
| `learning_rate` | 5e-5 | 5e-5 |
| `lr_scheduler_type` | cosine | cosine |
| `warmup_steps` | 第 1 轮 500，后续 0 | 500 |
| `weight_decay` | 0.05 | 0.05 |
| `fim_rate` | 0.5 | 0.5 |
| `fim_spm_rate` | 0.5 | 0.5 |
| `bf16` | True | True |

FIM 变换在 `ConstantLengthDataset` 中对已 tokenize 的序列随机切割重排，与数据来源无关，因此对自生成数据同样适用。

### 2.4 结果记录格式

`selfplay_results/results.csv`：

```csv
round,model_path,steps_total,humaneval_plus_pass1,mbpp_plus_pass1,livecodebench_pass1,timestamp
0,bigcode/santacoder,0,,,,
baseline_thestack,thestack_baseline/final_checkpoint,30000,,,,
selfplay_r1,selfplay_results/round1/final_checkpoint,6000,,,,
selfplay_r2,selfplay_results/round2/final_checkpoint,12000,,,,
selfplay_r3,selfplay_results/round3/final_checkpoint,18000,,,,
selfplay_r4,selfplay_results/round4/final_checkpoint,24000,,,,
selfplay_r5,selfplay_results/round5/final_checkpoint,30000,,,,
```

`steps_total` 使 self-play 曲线（0→6K→…→30K）和 baseline（30K）可在同一张图上展示。

---

## 三、环境搭建（从零开始）

### 3.1 硬件要求

- **GPU：** NVIDIA H200 或 A100（必须支持 BF16）
- **显存：** 推理 ~5GB，训练 ~20GB（开梯度检查点）
- **集群：** SLURM 调度

### 3.2 依赖版本与安装

**版本选择原则：** SantaCoder 使用自定义模型代码（`modeling_gpt2_mq.py`），其 KV cache 实现与 `transformers>=4.36` 引入的 `DynamicCache` 不兼容。锁定 `transformers==4.35.2`（DynamicCache 之前的最后版本）可一次性规避所有兼容性问题，无需任何 workaround。

```bash
# 创建项目目录
mkdir selfplay-finetune && cd selfplay-finetune

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# ---- 核心依赖（版本锁定） ----
pip install torch                          # 与 CUDA 版本匹配
pip install transformers==4.35.2           # 锁定版本，避免 KV cache 不兼容
pip install datasets>=2.14,<3.0            # 流式加载 + Arrow 支持
pip install accelerate>=0.24,<1.0          # HF Trainer 分布式训练
pip install wandb                          # 实验追踪（可禁用）

# ---- 评估框架 ----
pip install evalplus                       # EvalPlus (HumanEval+, MBPP+)

# LiveCodeBench
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench && pip install -e . && cd ..
```

**锁定 `transformers==4.35.2` 带来的好处：**

| 之前的问题 | 锁版本后 |
|-----------|---------|
| KV cache 格式不兼容，推理必须 `use_cache=False`（极慢） | KV cache 正常，`use_cache=True` 开箱即用 |
| 训练 checkpoint 的 `config.json` 中 `use_cache=False`，推理前需手动改 | 梯度检查点时自动关闭 cache，推理时正常启用，无需修改 |
| vLLM 无法加载 SantaCoder | vLLM 仍不可用（其自身不传 `trust_remote_code`），但 HF 推理因 cache 生效而足够快 |
| `generate_data.py` 推理每条约 60 秒（无 cache） | 推理每条约 5-10 秒（有 cache） |

### 3.3 关键环境变量

```bash
# 禁用 W&B（必须用 WANDB_MODE，不要用 WANDB_DISABLED）
# 原因：WANDB_DISABLED 已废弃，与 report_to='wandb' 同时存在时抛 RuntimeError
export WANDB_MODE=disabled

# 若 HuggingFace Hub 不可达（如内网服务器），强制离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 3.4 下载模型和数据

```bash
# 下载 SantaCoder 模型（约 4.5GB）
# 首次 from_pretrained 时自动下载到 ~/.cache/huggingface/hub/
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('bigcode/santacoder', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('bigcode/santacoder', trust_remote_code=True)
"

# The Stack 数据集（Python 子集）
# 流式加载，不需要提前下载（推荐，数据集 TB 级别）
# 若 Hub 不可达，需要本地 Arrow 缓存（见 §3.5）
```

### 3.5 离线环境：本地数据集缓存

若 HuggingFace Hub 不可达，需要使用本地 Arrow 缓存文件。缓存路径通常为：

```
~/.cache/huggingface/datasets/bigcode___the-stack-dedup/data.python/*/
```

**注意：** 该目录下会有 `cache-*.arrow` 索引文件（schema 为 `{indices: uint64}`），与实际数据文件的 schema 不兼容。加载时**必须**过滤掉：

```python
import glob, os
files = sorted([f for f in glob.glob(os.path.join(path, "*.arrow"))
                if not os.path.basename(f).startswith("cache-")])
```

---

## 四、目标项目结构

```
selfplay-finetune/
├── train.py                        # 核心训练脚本
├── fim.py                          # FIM 数据增强（PSM/SPM）
├── scripts/
│   ├── generate_data.py            # 批量推理，生成 self-play 训练数据
│   ├── run_selfplay_loop.sh        # 编排完整循环（生成→训练→评估）
│   ├── eval_evalplus.sh            # EvalPlus 评估脚本
│   ├── eval_livecodebench.sh       # LiveCodeBench 评估脚本
│   └── run_thestack_baseline.sh    # The Stack baseline 训练脚本
├── selfplay_results/               # self-play 实验输出
│   ├── generated_data/
│   │   ├── round1.jsonl
│   │   └── ...
│   ├── round1/final_checkpoint/
│   ├── ...
│   ├── results.csv
│   └── logs/
├── thestack_baseline/              # The Stack baseline 输出
│   └── final_checkpoint/
├── evalplus_results/               # EvalPlus 评估结果
├── livecodebench_results/          # LiveCodeBench 评估结果
└── LiveCodeBench/                  # 克隆的 LiveCodeBench 仓库
```

---

## 五、各文件完整实现规范

### 5.1 `fim.py` — FIM 数据增强

FIM（Fill-In-The-Middle）使模型能根据前后文预测中间缺失的代码。SantaCoder 预训练时使用了 FIM，其 tokenizer 内置了 FIM 特殊 token。Self-play 和 baseline 均使用 `fim_rate=0.5`。

```python
"""fim.py — Fill-In-The-Middle 数据增强

实现 FIM 论文 (Bavarian et al., 2022) 中的 PSM 和 SPM 两种变体。
FIM 变换在训练时对已 tokenize 的序列随机切割重排，使模型学会根据前后文预测中间内容。

PSM (Prefix-Suffix-Middle): <PRE> prefix <SUF> suffix <MID> middle
SPM (Suffix-Prefix-Middle): <PRE> <SUF> suffix <MID> prefix middle
"""
import functools
import numpy as np


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    """从 SantaCoder tokenizer 的 additional_special_tokens 中提取 FIM token IDs。

    SantaCoder 的 additional_special_tokens 按固定顺序存储 5 个 token，
    第 2-5 个依次为 FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD。
    """
    try:
        _, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = (
            tokenizer.special_tokens_map["additional_special_tokens"]
        )
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok]
            for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            None, None, None, None
        )
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


def permute(sample, np_rng, suffix_tok_id, prefix_tok_id, middle_tok_id,
            pad_tok_id, fim_rate, fim_spm_rate):
    """对 token 序列做 FIM 变换。

    以 fim_rate 概率触发变换；触发后以 fim_spm_rate 概率选择 SPM 或 PSM 格式。

    Args:
        sample: numpy array of token IDs
        np_rng: numpy RandomState
        fim_rate: 触发 FIM 变换的概率 (0.5 = 一半样本做 FIM)
        fim_spm_rate: FIM 变换中选择 SPM 格式的概率 (0.5 = SPM/PSM 各半)

    Returns:
        (new_sample, np_rng)
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = sample[: boundaries[0]]
        middle = sample[boundaries[0] : boundaries[1]]
        suffix = sample[boundaries[1] :]

        if np_rng.binomial(1, fim_spm_rate):
            # SPM: <PRE> <SUF> suffix <MID> prefix middle
            new_sample = np.concatenate(
                [[prefix_tok_id, suffix_tok_id], suffix,
                 [middle_tok_id], prefix, middle]
            )
        else:
            # PSM: <PRE> prefix <SUF> suffix <MID> middle
            new_sample = np.concatenate(
                [[prefix_tok_id], prefix, [suffix_tok_id],
                 suffix, [middle_tok_id], middle]
            )
        sample = new_sample

    return sample, np_rng
```

---

### 5.2 `train.py` — 核心训练脚本

**功能：** 从 HuggingFace Hub 或本地 JSONL 加载数据 → `ConstantLengthDataset` 打包为定长 2048-token 序列 → FIM 数据增强 → HuggingFace Trainer CLM 训练。

```python
"""train.py — SantaCoder 微调训练脚本

支持两种数据源：
  1. HuggingFace 数据集（The Stack 等），通过 --dataset_name 指定
  2. 本地 JSONL 文件（self-play 生成数据），通过 --local_data_path 指定

要求 transformers==4.35.2 以保证与 SantaCoder 自定义模型代码的兼容性。
"""
import argparse
import os

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

    # 数据集（二选一）
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-dedup")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集子目录，如 data/python")
    parser.add_argument("--data_column", type=str, default="content")
    parser.add_argument("--local_data_path", type=str, default=None,
                        help="本地 JSONL 文件路径。若设置，覆盖 --dataset_name")

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

    # FIM（SantaCoder 预训练使用了 FIM，微调时也应开启）
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

    # 梯度检查点（默认开启以节省显存）
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    return parser.parse_args()


# ============================================================
# 工具函数
# ============================================================
def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """采样 400 条估算字符/token 比率（SantaCoder Python 约 3~4）。"""
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

    def __init__(self, tokenizer, dataset, seq_length=2048,
                 num_of_sequences=1024, chars_per_token=3.6,
                 content_field="content", fim_rate=0.5, fim_spm_rate=0.5,
                 seed=0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
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

            np_rng.shuffle(examples)

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
        dataset = load_dataset("json",
                               data_files={"train": args.local_data_path},
                               split="train")
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
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
        tokenizer, train_data, seq_length=args.seq_length,
        content_field=args.data_column, chars_per_token=chars_per_token,
        fim_rate=args.fim_rate, fim_spm_rate=args.fim_spm_rate, seed=args.seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, seq_length=args.seq_length,
        content_field=args.data_column, chars_per_token=chars_per_token,
        fim_rate=0, seed=args.seed,  # 验证集不做 FIM
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
        eval_strategy="steps",
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
    trainer.save_model(os.path.join(args.output_dir, "final_checkpoint"))
    print(f"训练完成，最终 checkpoint 保存至 {args.output_dir}/final_checkpoint/")


if __name__ == "__main__":
    args = get_args()
    run_training(args)
```

#### SantaCoder 注意事项

- **`trust_remote_code=True`** 是必须的。SantaCoder 使用自定义模型代码（`GPT2CustomConfig` + `modeling_gpt2_mq.py`），不传此参数会加载失败。
- **BF16 而非 FP16。** FP16 在小 batch 下容易梯度 overflow（`grad_norm: nan`）。
- **梯度检查点默认开启。** 训练时自动 `use_cache=False`（与 KV cache 不兼容），推理时恢复。因为锁定了 `transformers==4.35.2`，无需手动处理。

---

### 5.3 `scripts/generate_data.py` — Self-play 数据生成

从 The Stack Python 流式加载，取每个文件前 1024 tokens 作 prompt，用模型生成后续 1024 tokens，保存为训练兼容的 JSONL。

因锁定 `transformers==4.35.2`，KV cache 正常工作，推理速度有保障。

```python
"""scripts/generate_data.py — Self-play 数据生成脚本

用法：
    python scripts/generate_data.py \
        --model_path bigcode/santacoder \
        --output_file selfplay_results/generated_data/round1.jsonl \
        --num_samples 5000 --seed 1

要求 transformers==4.35.2。
"""
import argparse
import json
import os
import glob

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
        device_map="auto",
    )
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
                    # transformers==4.35.2 下 KV cache 正常工作，无需 use_cache=False
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
```

#### Tokenizer 注意事项

SantaCoder tokenizer 没有 `pad_token`，batch 推理前必须设置：
```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # causal LM 生成要求左侧 padding
```

---

### 5.4 `scripts/eval_evalplus.sh` — EvalPlus 评估

```bash
#!/bin/bash
# EvalPlus 评估（HumanEval+ 和 MBPP+）
# 用法：sbatch scripts/eval_evalplus.sh <model_path> [humaneval|mbpp]
# 默认同时评估 humaneval 和 mbpp

#SBATCH --job-name=eval_evalplus
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=selfplay_results/logs/evalplus_%j.out

set -euo pipefail

MODEL_PATH="${1:?用法: $0 <model_path> [humaneval|mbpp]}"
DATASETS="${2:-humaneval mbpp}"
BACKEND="hf"
RESULTS_DIR="./evalplus_results"

mkdir -p "$RESULTS_DIR"

for DATASET in $DATASETS; do
    echo "===== 评估 $DATASET ====="
    evalplus.evaluate \
        --model "$MODEL_PATH" \
        --dataset "$DATASET" \
        --backend "$BACKEND" \
        --greedy \
        --trust-remote-code \
        --output-dir "$RESULTS_DIR"
    echo "===== $DATASET 完成 ====="
done
```

**为什么用 `--backend hf` 而不是 vLLM：** vLLM 内部用 `AutoModel.from_config()` 不传 `trust_remote_code`，无法识别 SantaCoder 的自定义 `GPT2CustomConfig`。HF backend 在 `transformers==4.35.2` + KV cache 正常的情况下速度足够。

---

### 5.5 `scripts/eval_livecodebench.sh` — LiveCodeBench 评估

```bash
#!/bin/bash
# LiveCodeBench 评估
# 用法：sbatch scripts/eval_livecodebench.sh <model_path> [scenario] [version]
# 默认 codegeneration, release_latest

#SBATCH --job-name=eval_lcb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=selfplay_results/logs/lcb_%j.out

set -euo pipefail

MODEL_PATH="${1:?用法: $0 <model_path> [scenario] [version]}"
SCENARIO="${2:-codegeneration}"
VERSION="${3:-release_latest}"
RESULTS_DIR="./livecodebench_results"

mkdir -p "$RESULTS_DIR"

echo "===== LiveCodeBench: $SCENARIO ($VERSION) ====="
python -m lcb_runner.runner.main \
    --model "$MODEL_PATH" \
    --scenario "$SCENARIO" \
    --release_version "$VERSION" \
    --trust_remote_code \
    --output_dir "$RESULTS_DIR"
echo "===== 完成 ====="
```

---

### 5.6 `scripts/run_thestack_baseline.sh` — The Stack Baseline 训练

```bash
#!/bin/bash
# The Stack baseline（30K 步，Python，FIM 0.5）— self-play 的参照上限

#SBATCH --job-name=thestack_baseline
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=selfplay_results/logs/baseline_%j.out

set -euo pipefail

WANDB_MODE=disabled torchrun --nproc_per_node 1 --standalone train.py \
    --dataset_name bigcode/the-stack-dedup \
    --data_dir data/python \
    --data_column content \
    --streaming \
    --seq_length 2048 \
    --max_steps 30000 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --weight_decay 0.05 \
    --fim_rate 0.5 \
    --fim_spm_rate 0.5 \
    --bf16 \
    --output_dir thestack_baseline \
    --eval_freq 3000 \
    --save_freq 3000 \
    --log_freq 100

echo "Baseline 训练完成，开始评估..."
bash scripts/eval_evalplus.sh thestack_baseline/final_checkpoint
bash scripts/eval_livecodebench.sh thestack_baseline/final_checkpoint codegeneration release_latest
```

---

### 5.7 `scripts/run_selfplay_loop.sh` — Self-play 完整循环

```bash
#!/bin/bash
# Self-Play 循环：Round 0 评估 → 5 轮（生成→训练→评估）
# 用法：sbatch scripts/run_selfplay_loop.sh

#SBATCH --job-name=selfplay_loop
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=selfplay_results/logs/loop_%j.out

set -euo pipefail

# ---- 配置 ----
MODEL_PATH="bigcode/santacoder"
TOTAL_ROUNDS=5
STEPS_PER_ROUND=6000
NUM_SAMPLES=5000
SHUFFLE_BUFFER=50000

# 若 HF Hub 不可达，取消注释并填写本地缓存路径
# LOCAL_DATASET_PATH="$HOME/.cache/huggingface/datasets/bigcode___the-stack-dedup/data.python/..."

# ---- 初始化 ----
mkdir -p selfplay_results/generated_data selfplay_results/logs

CSV="selfplay_results/results.csv"
if [ ! -f "$CSV" ]; then
    echo "round,model_path,steps_total,humaneval_plus_pass1,mbpp_plus_pass1,livecodebench_pass1,timestamp" > "$CSV"
fi

# ---- Round 0：评估原始模型 ----
echo "========== Round 0: 评估原始 SantaCoder =========="
bash scripts/eval_evalplus.sh "$MODEL_PATH"
bash scripts/eval_livecodebench.sh "$MODEL_PATH" codegeneration release_latest
echo "0,$MODEL_PATH,0,,,,$(date -Iseconds)" >> "$CSV"

# ---- Self-Play 循环 ----
for ROUND in $(seq 1 $TOTAL_ROUNDS); do
    PREV_ROUND=$((ROUND - 1))

    if [ $ROUND -eq 1 ]; then
        CURRENT_MODEL="$MODEL_PATH"
        WARMUP=500
    else
        CURRENT_MODEL="selfplay_results/round${PREV_ROUND}/final_checkpoint"
        WARMUP=0
    fi

    echo "========== Round $ROUND / $TOTAL_ROUNDS =========="

    # Step 1: 生成数据
    echo "--- Step 1: 生成数据 ---"
    GEN_CMD="python scripts/generate_data.py \
        --model_path $CURRENT_MODEL \
        --output_file selfplay_results/generated_data/round${ROUND}.jsonl \
        --num_samples $NUM_SAMPLES \
        --shuffle_buffer $SHUFFLE_BUFFER \
        --seed $ROUND"

    if [ -n "${LOCAL_DATASET_PATH:-}" ]; then
        GEN_CMD="$GEN_CMD --local_dataset_path $LOCAL_DATASET_PATH"
    fi
    eval $GEN_CMD

    LINES=$(wc -l < "selfplay_results/generated_data/round${ROUND}.jsonl")
    echo "生成了 $LINES 条样本"

    # Step 2: 训练（FIM 开启，与 baseline 一致）
    echo "--- Step 2: 训练 ---"
    WANDB_MODE=disabled torchrun --nproc_per_node 1 --standalone train.py \
        --local_data_path "selfplay_results/generated_data/round${ROUND}.jsonl" \
        --model_path "$CURRENT_MODEL" \
        --seq_length 2048 \
        --max_steps $STEPS_PER_ROUND \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --warmup_steps $WARMUP \
        --weight_decay 0.05 \
        --fim_rate 0.5 \
        --fim_spm_rate 0.5 \
        --bf16 \
        --output_dir "selfplay_results/round${ROUND}" \
        --eval_freq 1000 \
        --save_freq $STEPS_PER_ROUND \
        --log_freq 100

    # Step 3: 评估（EvalPlus + LiveCodeBench）
    echo "--- Step 3: 评估 ---"
    CKPT="selfplay_results/round${ROUND}/final_checkpoint"
    bash scripts/eval_evalplus.sh "$CKPT"
    bash scripts/eval_livecodebench.sh "$CKPT" codegeneration release_latest

    # Step 4: 记录
    STEPS_TOTAL=$((ROUND * STEPS_PER_ROUND))
    echo "selfplay_r${ROUND},$CKPT,$STEPS_TOTAL,,,,$(date -Iseconds)" >> "$CSV"

    echo "========== Round $ROUND 完成（累计 $STEPS_TOTAL 步）=========="
done

echo ""
echo "===== Self-Play 实验完成 ====="
cat "$CSV"
```

---

## 六、验证检查点

在正式跑完整循环前，按以下顺序验证各组件：

### 验证 1：数据生成

```bash
python scripts/generate_data.py \
    --model_path bigcode/santacoder \
    --output_file /tmp/test_gen.jsonl \
    --num_samples 5 --batch_size 1

wc -l /tmp/test_gen.jsonl                                                    # 应为 5
python -c "import json; [json.loads(l) for l in open('/tmp/test_gen.jsonl')]" # 无报错
```

**通过标准：** 5 条 JSONL，每条 `content` 1000–10000 字符。

### 验证 2：本地 JSONL 训练（含 FIM）

```bash
WANDB_MODE=disabled torchrun --nproc_per_node 1 --standalone train.py \
    --local_data_path /tmp/test_gen.jsonl \
    --model_path bigcode/santacoder \
    --seq_length 2048 --max_steps 50 \
    --batch_size 1 --gradient_accumulation_steps 2 \
    --fim_rate 0.5 --fim_spm_rate 0.5 --bf16 \
    --output_dir /tmp/test_ckpt \
    --eval_freq 25 --save_freq 50 --log_freq 5

ls /tmp/test_ckpt/final_checkpoint/   # 应存在 config.json, model.safetensors 等
```

**通过标准：** 50 步完成，loss 下降，`final_checkpoint/` 存在。

### 验证 3：EvalPlus

```bash
bash scripts/eval_evalplus.sh bigcode/santacoder humaneval
```

### 验证 4：LiveCodeBench

```bash
bash scripts/eval_livecodebench.sh bigcode/santacoder codegeneration release_latest
```

---

## 七、可期待的实验结论

实验结束后，根据 self-play 曲线（x 轴 `steps_total`，y 轴 `humaneval_plus_pass1` / `mbpp_plus_pass1` / `livecodebench_pass1`）的走势，可以支持以下之一的论点：

1. **持续提升但有上限** — self-play 前几轮有正向效果，最终收敛到低于 baseline → 数据质量是瓶颈，但迭代有帮助
2. **早期提升后退化（mode collapse）** — 1–2 轮性能略升后持续下降 → 模型在强化自身偏见，需要真实数据纠正
3. **无明显变化** — self-play 对性能几乎没有影响 → 模型在自身数据上没有新的学习信号

任何结果都是有价值的 finding。实验设计确保 self-play 与 baseline 的**唯一差异为数据来源**（超参数、FIM、总步数完全一致），结论可归因。

---

## 八、快速上手顺序

```
 1. 环境搭建：安装依赖（§3.2），设环境变量（§3.3）
 2. 下载模型和数据（§3.4）
 3. 创建 fim.py（§5.1）
 4. 创建 train.py（§5.2）
 5. 创建 scripts/generate_data.py（§5.3）
 6. 创建评估脚本（§5.4, §5.5）
 7. 验证 1-4（§六）
 8. 提交 The Stack baseline 训练 + 评估（§5.6）
 9. 提交 self-play 循环（§5.7）
10. 填写 results.csv，画图
```

---

*本文档包含从零重建项目所需的全部信息。所有代码为完整实现，不依赖任何已有代码库。锁定 `transformers==4.35.2` 规避了 SantaCoder 与新版 transformers 的所有已知兼容性问题。*
