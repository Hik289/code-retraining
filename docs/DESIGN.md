# Multi-Model Self-Play Experiment Design (V2)

> 目标：将 self-play fine-tuning + data filtering 实验从单模型 (SantaCoder) 扩展到 4 个代码生成模型，观察 model collapse 现象在不同模型规模/架构下的表现差异。
>
> 前置工作：V1 实验已在 SantaCoder 上完成 5 组实验（no-filter + 4 种 filter），结论：所有 filter 均延缓但不阻止 collapse。详见 `old/progress.md`。
>
> 日期：2026-04-14

---

## 一、研究目标

**核心问题不变：** 用模型自己生成的数据迭代训练自己，性能如何变化？Data filtering 能否缓解退化？

**V2 新增问题：**
1. **模型规模效应** — 1.1B / 1.5B / 3B / 7B 的 collapse 速率是否不同？更大的模型是否更抗 collapse？
2. **架构/预训练数据效应** — 同量级下不同预训练背景的模型（SantaCoder vs Qwen2.5-Coder）collapse 模式是否相同？
3. **Filter 通用性** — V1 中 compile filter 对 SantaCoder 最有效，这个结论是否跨模型成立？
4. **Self-referential scoring 的普遍性** — V1 发现 binary classifier 自评分随 collapse 失真，这在其他模型上是否同样发生？

---

## 二、模型矩阵

| 模型 | HuggingFace ID | 参数量 | 上下文长度 | 预训练 FIM | transformers 版本要求 |
|------|---------------|-------|-----------|-----------|---------------------|
| SantaCoder | `bigcode/santacoder` | 1.1B | 2048 | 是 | `==4.35.2`（KV cache 兼容性） |
| StarCoder2-3B | `bigcode/starcoder2-3b` | 3B | 16384 | 是 | `>=4.39` |
| Qwen2.5-Coder-1.5B | `Qwen/Qwen2.5-Coder-1.5B` | 1.5B | 32768 | 是 | `>=4.37` |
| Code Llama 7B | `codellama/CodeLlama-7b-hf` | 7B | 16384 | 是 | `>=4.34` |

### 模型选择理由

- **SantaCoder (1.1B)**: V1 基线，BigCode 项目第一代，预训练于 Python/Java/JS
- **StarCoder2-3B**: BigCode 第二代，预训练语料更大更干净（The Stack v2），架构更现代
- **Qwen2.5-Coder-1.5B**: 非 BigCode 系列，阿里团队，不同预训练策略/数据，与 SantaCoder 同量级可直接对比
- **Code Llama 7B**: Meta 出品，基于 Llama 2 继续预训练，7B 级别观察更大模型的 collapse 行为

### FIM Token 格式

| 模型 | PREFIX | MIDDLE | SUFFIX | PAD |
|------|--------|--------|--------|-----|
| SantaCoder | `<fim-prefix>` | `<fim-middle>` | `<fim-suffix>` | `<fim-pad>` |
| StarCoder2-3B | `<fim_prefix>` | `<fim_middle>` | `<fim_suffix>` | `<fim_pad>` |
| Qwen2.5-Coder | `<|fim_prefix|>` | `<|fim_middle|>` | `<|fim_suffix|>` | `<|fim_pad|>` |
| Code Llama | `▁<PRE>` | `▁<MID>` | `▁<SUF>` | N/A |

> **注意**：Code Llama 的 FIM token 在 tokenizer 中以 sentencepiece 编码存储（前缀 `▁` 为 sp 空格标记）。实际使用时通过 `tokenizer.special_tokens_map` 或直接查 `vocab` 获取 ID。

### transformers 版本隔离

| 模型 | transformers 要求 | venv 路径 |
|------|-------------------|-----------|
| SantaCoder | `==4.35.2` | `venvs/santacoder/` |
| StarCoder2-3B | `>=4.39` | `venvs/starcoder2/` |
| Qwen2.5-Coder-1.5B | `>=4.37` | `venvs/qwen25/` |
| Code Llama 7B | `>=4.34` | `venvs/codellama/` |

每个 venv 独立安装 `transformers + torch + datasets + evalplus + tqdm` 等依赖，避免版本冲突。SantaCoder 的 KV cache 在 transformers>=4.36 上报错，必须 pin 到 4.35.2。

---

## 三、实验矩阵

### 完整矩阵：4 模型 × 5 过滤 = 20 组实验

| # | 模型 | Filter | 实验 ID | 输出目录 |
|---|------|--------|---------|---------|
| 1 | SantaCoder | No Filter | `santa_nofilter` | `results/santacoder/no_filter/` |
| 2 | SantaCoder | Compile | `santa_compile` | `results/santacoder/compile_filter/` |
| 3 | SantaCoder | Compile+Quality | `santa_quality` | `results/santacoder/quality_filter/` |
| 4 | SantaCoder | PPL | `santa_ppl` | `results/santacoder/ppl_filter/` |
| 5 | SantaCoder | Binary Classifier | `santa_binary` | `results/santacoder/binary_filter/` |
| 6 | StarCoder2-3B | No Filter | `sc2_nofilter` | `results/starcoder2/no_filter/` |
| 7 | StarCoder2-3B | Compile | `sc2_compile` | `results/starcoder2/compile_filter/` |
| 8 | StarCoder2-3B | Compile+Quality | `sc2_quality` | `results/starcoder2/quality_filter/` |
| 9 | StarCoder2-3B | PPL | `sc2_ppl` | `results/starcoder2/ppl_filter/` |
| 10 | StarCoder2-3B | Binary Classifier | `sc2_binary` | `results/starcoder2/binary_filter/` |
| 11 | Qwen2.5-Coder-1.5B | No Filter | `qwen_nofilter` | `results/qwen25/no_filter/` |
| 12 | Qwen2.5-Coder-1.5B | Compile | `qwen_compile` | `results/qwen25/compile_filter/` |
| 13 | Qwen2.5-Coder-1.5B | Compile+Quality | `qwen_quality` | `results/qwen25/quality_filter/` |
| 14 | Qwen2.5-Coder-1.5B | PPL | `qwen_ppl` | `results/qwen25/ppl_filter/` |
| 15 | Qwen2.5-Coder-1.5B | Binary Classifier | `qwen_binary` | `results/qwen25/binary_filter/` |
| 16 | Code Llama 7B | No Filter | `llama_nofilter` | `results/codellama/no_filter/` |
| 17 | Code Llama 7B | Compile | `llama_compile` | `results/codellama/compile_filter/` |
| 18 | Code Llama 7B | Compile+Quality | `llama_quality` | `results/codellama/quality_filter/` |
| 19 | Code Llama 7B | PPL | `llama_ppl` | `results/codellama/ppl_filter/` |
| 20 | Code Llama 7B | Binary Classifier | `llama_binary` | `results/codellama/binary_filter/` |

### 统一训练超参数

所有 20 组实验使用相同的训练配置，唯一变量是模型和过滤策略：

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_steps` | 3000 / 轮 | 每轮训练步数 |
| `total_rounds` | 20 | 总轮次 |
| `num_samples` | 5000 | 每轮训练样本数 |
| `batch_size` | 2 | per-device batch size |
| `gradient_accumulation_steps` | 4 | 有效 batch = 2 × 4 = 8 |
| `learning_rate` | 5e-5 | |
| `lr_scheduler_type` | cosine | |
| `warmup_steps` | 500（仅 R1） | 后续轮次 warmup=0 |
| `weight_decay` | 0.05 | |
| `seq_length` | 2048 | 所有模型统一截断到 2048（SantaCoder 最大上下文） |
| `fim_rate` | 0.5 | FIM 变换概率 |
| `fim_spm_rate` | 0.5 | SPM/PSM 各 50% |
| `bf16` | True | |
| `temperature` | 0.8 | 生成时 |
| `top_p` | 0.95 | 生成时 |
| `prompt_tokens` | 1024 | 从 The Stack 取前 1024 token 作 prompt |
| `max_new_tokens` | 1024 | 生成 1024 token |

> **seq_length=2048 说明**：虽然 StarCoder2/Qwen/Code Llama 支持更长上下文，但为了横向公平对比，统一使用 SantaCoder 的最大长度 2048。这也让所有模型的计算成本可比。

### 过滤策略参数

| Filter | 关键参数 | 数据量策略 |
|--------|---------|-----------|
| No Filter | — | 一次生成 5000 条 |
| Compile | `compile()` 语法检查 | 循环生成直到过滤后满 5000 条 |
| Compile+Quality | repetition_threshold=0.5, min_completion_tokens=50 | 循环生成直到过滤后满 5000 条 |
| PPL | 保留 PPL 最低的 top 25% | 一次生成 20000 条 → 打分 → 取 top 5000 |
| Binary Classifier | 保留 score 最高的 top 25% | 一次生成 20000 条 → 打分 → 取 top 5000 |

### 评估

每轮训练后评估三个 benchmark：

| Benchmark | 工具 | 指标 |
|-----------|------|------|
| HumanEval / HumanEval+ | EvalPlus | pass@1 (greedy, temp=0.2) |
| MBPP / MBPP+ | EvalPlus | pass@1 (greedy, temp=0.2) |
| LiveCodeBench (release_v1) | 自定义脚本 | pass@1 |

### Experiment Tracking

#### 统一 CSV Schema

所有 20 组实验的 `results.csv` 使用完全相同的列定义，方便后续一行 `pd.concat` 汇总：

```
model,filter,round,steps_total,humaneval_pass1,humaneval_plus_pass1,mbpp_pass1,mbpp_plus_pass1,livecodebench_pass1,train_loss,num_generated,num_after_filter,filter_pass_rate,generation_time_sec,training_time_sec,eval_time_sec,timestamp
```

| 列名 | 类型 | 说明 |
|------|------|------|
| `model` | str | 模型短名：`santacoder` / `starcoder2` / `qwen25` / `codellama` |
| `filter` | str | 过滤策略：`none` / `compile` / `quality` / `ppl` / `binary` |
| `round` | int | 轮次（1-20） |
| `steps_total` | int | 累计训练步数 |
| `humaneval_pass1` | float | HumanEval base pass@1 |
| `humaneval_plus_pass1` | float | HumanEval+ pass@1 |
| `mbpp_pass1` | float | MBPP base pass@1 |
| `mbpp_plus_pass1` | float | MBPP+ pass@1 |
| `livecodebench_pass1` | float | LiveCodeBench pass@1 |
| `train_loss` | float | 训练最终 loss（从 trainer log 提取） |
| `num_generated` | int | 本轮总生成数量（过滤前） |
| `num_after_filter` | int | 过滤后实际用于训练的数量 |
| `filter_pass_rate` | float | `num_after_filter / num_generated` |
| `generation_time_sec` | int | 数据生成+过滤耗时（秒） |
| `training_time_sec` | int | 训练耗时（秒） |
| `eval_time_sec` | int | 评估耗时（秒） |
| `timestamp` | str | ISO 8601 时间戳 |

> **关键设计**：`model` 和 `filter` 列冗余写入每行，这样单个 CSV 自包含，合并后无需靠文件路径判断来源。

#### 汇总机制

`src/aggregate_results.py`：扫描 `results/*/*/results.csv`，合并为一张总表 + 自动生成对比图。

```bash
python src/aggregate_results.py --output_dir results/summary/

# 输出：
# results/summary/
#   all_results.csv          — 20 组 CSV 合并（~400 行 = 20 实验 × 20 轮）
#   humaneval_by_model.png   — 4 条线（4 模型），x=round, y=humaneval_pass1, 按 filter 分面
#   mbpp_by_model.png        — 同上
#   collapse_speed.csv       — 每组实验降至 baseline 50% 的轮次
#   filter_passrate.png      — 过滤通过率随轮次变化（观察 collapse 对生成质量的影响）
```

#### 每轮 metadata JSON（可选）

除了 CSV 汇总行，每轮结束后写一个 `round{N}_meta.json`：

```json
{
  "model": "starcoder2",
  "filter": "compile",
  "round": 3,
  "model_path": "results/starcoder2/compile_filter/round2/final_checkpoint",
  "generated_data": "results/starcoder2/compile_filter/generated_data/round3.jsonl",
  "num_generated": 7200,
  "num_after_filter": 5000,
  "filter_pass_rate": 0.694,
  "train_loss": 2.31,
  "eval": {
    "humaneval_pass1": 0.152,
    "humaneval_plus_pass1": 0.134,
    "mbpp_pass1": 0.285,
    "mbpp_plus_pass1": 0.241,
    "livecodebench_pass1": 0.012
  },
  "timing": {
    "generation_sec": 1820,
    "training_sec": 3600,
    "eval_sec": 900
  },
  "timestamp": "2026-04-16T14:23:00+00:00"
}
```

存储路径：`results/{model}/{filter}/round{N}_meta.json`。CSV 是给汇总用的，JSON 是给单轮调试用的。

---

## 四、新 Codebase 设计

> **原则**：所有新代码放在 `src/` 目录下，从零编写，不直接依赖旧文件。旧代码（`train.py`, `fim.py`, `scripts/`）保留在原位作为只读参考。

### 目录结构

```
self-replay/
├── docs/DESIGN.md             # 本文档
│
├── src/                       # ====== 所有 V2 新代码 ======
│   ├── config.py              # 模型配置加载（YAML → dict）
│   ├── fim.py                 # FIM 变换（多模型）
│   ├── train.py               # 训练脚本（多模型 + 多版本 transformers）
│   ├── generate.py            # 数据生成（The Stack prompt → model completion）
│   ├── filters.py             # 所有过滤逻辑（compile / quality / ppl / binary）
│   ├── evaluate_evalplus.py   # EvalPlus 代码生成 + 调用评估
│   ├── evaluate_lcb.py        # LiveCodeBench 代码生成 + 评估
│   ├── aggregate_results.py   # 汇总 20 组 results.csv + 生成对比图
│   ├── run_experiment.sh      # SLURM 统一实验入口
│   └── setup_venvs.sh         # 一键创建所有 venv
│
├── configs/                   # 模型配置
│   ├── santacoder.yaml
│   ├── starcoder2.yaml
│   ├── qwen25.yaml
│   └── codellama.yaml
│
├── venvs/                     # 隔离虚拟环境（per-model）
│   ├── santacoder/
│   ├── starcoder2/
│   ├── qwen25/
│   └── codellama/
│
├── results/                   # V2 实验结果（按 model × filter 组织）
│   ├── santacoder/
│   │   ├── no_filter/
│   │   │   ├── generated_data/    # round1.jsonl, round2.jsonl, ...
│   │   │   ├── round1/            # final_checkpoint/
│   │   │   ├── round2/
│   │   │   └── results.csv        # 评估指标汇总
│   │   ├── compile_filter/
│   │   ├── quality_filter/
│   │   ├── ppl_filter/
│   │   └── binary_filter/
│   ├── starcoder2/
│   │   └── ...（同结构）
│   ├── qwen25/
│   │   └── ...
│   └── codellama/
│       └── ...
│
├── evalplus_results/          # EvalPlus 生成文件（临时）
├── livecodebench_results/     # LCB 生成+评估文件（临时）
│
│   # ====== V1 旧代码（只读参考，不运行） ======
├── train.py                   # V1 训练脚本
├── fim.py                     # V1 FIM
├── scripts/                   # V1 脚本
│   ├── generate_data.py
│   ├── generate_data_filtered.py
│   ├── score_perplexity.py
│   ├── score_binary_classifier.py
│   ├── filter_perplexity.py
│   ├── filter_binary_classifier.py
│   ├── evalplus_generate.py
│   ├── lcb_generate.py
│   └── run_selfplay_*.sh
└── selfplay_results/          # V1 实验结果（只读参考）
```

### src/config.py — 模型配置系统

加载 `configs/*.yaml`，提供统一接口：

```python
# 用法
from src.config import load_model_config

cfg = load_model_config("santacoder")  # 读取 configs/santacoder.yaml
print(cfg["model_id"])          # "bigcode/santacoder"
print(cfg["fim_prefix"])        # "<fim-prefix>"
print(cfg["trust_remote_code"]) # True
```

配置文件示例（`configs/santacoder.yaml`）：

```yaml
model_id: "bigcode/santacoder"
short_name: "santacoder"
params: "1.1B"
max_context: 2048
trust_remote_code: true

# FIM tokens（直接写 token 字面量，config.py 转为 ID）
fim_prefix: "<fim-prefix>"
fim_middle: "<fim-middle>"
fim_suffix: "<fim-suffix>"
fim_pad: "<fim-pad>"

# Binary classifier tokens
binary_good_token: " good"
binary_bad_token: " bad"

# 评估停止序列
stop_sequences_humaneval: ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"]
stop_sequences_mbpp: ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"]
```

### src/fim.py — FIM 变换

两个函数：

```python
def get_fim_token_ids(tokenizer, config):
    """根据配置获取 FIM token IDs。支持所有 4 个模型。"""
    prefix_id = tokenizer.convert_tokens_to_ids(config["fim_prefix"])
    middle_id = tokenizer.convert_tokens_to_ids(config["fim_middle"])
    suffix_id = tokenizer.convert_tokens_to_ids(config["fim_suffix"])
    pad_id    = tokenizer.convert_tokens_to_ids(config["fim_pad"])
    return suffix_id, prefix_id, middle_id, pad_id

def permute(sample, np_rng, suffix_tok_id, prefix_tok_id, middle_tok_id,
            pad_tok_id, fim_rate=0.5, fim_spm_rate=0.5):
    """FIM 变换（PSM/SPM）。与旧版逻辑完全相同，model-agnostic。"""
    ...
```

### src/train.py — 训练

核心组件：
- **`ConstantLengthDataset`**：定长序列打包，`infinite=True` 循环小数据集，集成 FIM
- **多版本 transformers 兼容**：检测版本自动选择 `evaluation_strategy` vs `eval_strategy`
- **模型加载**：通过 `--config` 参数读取 YAML，`trust_remote_code` 等从配置获取

```bash
# 用法
python src/train.py \
    --config configs/starcoder2.yaml \
    --local_data_path results/starcoder2/compile_filter/generated_data/round1.jsonl \
    --max_steps 3000 \
    --output_dir results/starcoder2/compile_filter/round1
```

### src/generate.py — 数据生成

从 The Stack 取 prompt → 模型生成 completion → 输出 JSONL。

支持两种模式：
1. **无过滤**：一次生成 N 条，直接输出
2. **带过滤**：循环生成 + 在线过滤，直到过滤后达到目标数量

```bash
# 无过滤，生成 5000 条
python src/generate.py \
    --config configs/qwen25.yaml \
    --model_path Qwen/Qwen2.5-Coder-1.5B \
    --output_file results/qwen25/no_filter/generated_data/round1.jsonl \
    --num_samples 5000

# 带 compile 过滤，循环直到 5000 条
python src/generate.py \
    --config configs/qwen25.yaml \
    --model_path Qwen/Qwen2.5-Coder-1.5B \
    --output_file results/qwen25/compile_filter/generated_data/round1.jsonl \
    --num_samples 5000 \
    --filter_mode compile

# 为 PPL/Binary 过滤生成 20000 条原始数据
python src/generate.py \
    --config configs/qwen25.yaml \
    --model_path Qwen/Qwen2.5-Coder-1.5B \
    --output_file results/qwen25/ppl_filter/generated_data/round1_raw.jsonl \
    --num_samples 20000
```

### src/filters.py — 过滤逻辑

所有过滤功能整合在一个文件中，通过子命令调用：

```bash
# compile 检查（纯 CPU）
python src/filters.py compile \
    --input_file round1_raw.jsonl \
    --output_file round1.jsonl

# compile + quality（纯 CPU）
python src/filters.py quality \
    --input_file round1_raw.jsonl \
    --output_file round1.jsonl \
    --repetition_threshold 0.5 \
    --min_completion_tokens 50

# PPL 打分（需 GPU）
python src/filters.py score-ppl \
    --input_file round1_raw.jsonl \
    --output_file round1_scored.jsonl \
    --model_path results/starcoder2/ppl_filter/round0/final_checkpoint

# PPL 过滤（纯 CPU）
python src/filters.py filter-topk \
    --input_file round1_scored.jsonl \
    --output_file round1.jsonl \
    --score_field ppl \
    --top_percent 25 \
    --ascending  # PPL 越低越好

# Binary 打分（需 GPU）
python src/filters.py score-binary \
    --input_file round1_raw.jsonl \
    --output_file round1_scored.jsonl \
    --config configs/starcoder2.yaml \
    --model_path results/starcoder2/binary_filter/round0/final_checkpoint

# Binary 过滤（纯 CPU）
python src/filters.py filter-topk \
    --input_file round1_scored.jsonl \
    --output_file round1.jsonl \
    --score_field score \
    --top_percent 25
    # 默认 descending：score 越高越好
```

### src/evaluate_evalplus.py & src/evaluate_lcb.py — 评估

```bash
# EvalPlus
python src/evaluate_evalplus.py \
    --config configs/starcoder2.yaml \
    --model_path results/starcoder2/compile_filter/round1/final_checkpoint \
    --dataset humaneval \
    --output_file evalplus_results/humaneval/sc2_compile_r1.jsonl

# LiveCodeBench
python src/evaluate_lcb.py \
    --config configs/starcoder2.yaml \
    --model_path results/starcoder2/compile_filter/round1/final_checkpoint \
    --output_dir livecodebench_results
```

### src/run_experiment.sh — SLURM 统一入口

```bash
# 用法
sbatch src/run_experiment.sh santacoder compile
sbatch src/run_experiment.sh starcoder2 ppl
sbatch src/run_experiment.sh codellama none

# 内部流程
# 1. 读取 configs/${MODEL}.yaml
# 2. 激活 venvs/${MODEL}/bin/activate
# 3. for ROUND in 1..20:
#      a. 数据生成（根据 filter 选择流程）
#      b. python src/train.py --config ... --local_data_path ...
#      c. python src/evaluate_evalplus.py ... (HumanEval + MBPP)
#      d. python src/evaluate_lcb.py ...
#      e. 追加结果到 results.csv
```

### 旧代码的角色

旧代码（`train.py`, `fim.py`, `scripts/`）保留在原位，但 V2 实验**不运行任何旧代码**。旧代码的价值：
- **只读参考**：编写 `src/` 新代码时参考旧实现的算法逻辑
- **V1 可复现**：如果需要重跑 V1 实验，旧脚本仍可用
- **对比**：`selfplay_results/` 中的 V1 结果保留，用于与 V2 SantaCoder 结果交叉验证

---

## 五、实现计划

> **开发原则**：Bottom-up。先写每个子步骤的独立脚本，逐个用小数据验证，然后组合成完整 loop，最后 smoke test 全流程再正式提交。

### Phase 1: 基础设施 + 配置

| 步骤 | 产出 | 验证方法 |
|------|------|---------|
| 1a | `src/config.py` | 加载 4 个 YAML，打印配置，确认无报错 |
| 1b | `configs/*.yaml` （4 个） | 被 1a 加载验证 |
| 1c | `src/setup_venvs.sh` | 创建 4 个 venv，各自 `python -c "import transformers; print(transformers.__version__)"` |
| 1d | 验证模型加载 | 各 venv 中 `from_pretrained` 加载模型，打印 FIM token IDs、`" good"`/`" bad"` token IDs |

### Phase 2: 子步骤脚本（逐个编写 + 小数据测试）

每个脚本写完后立即用小规模数据验证：

| 步骤 | 脚本 | 小测验证 |
|------|------|---------|
| 2a | `src/fim.py` | 对 4 个模型各跑一次 `get_fim_token_ids()`，验证 ID 非 None；用 dummy tokens 跑 `permute()` |
| 2b | `src/generate.py` | 每个模型生成 **20 条**，检查输出 JSONL 格式正确、content 非空 |
| 2c | `src/filters.py` — compile 子命令 | 用 2b 的 20 条跑 compile 过滤，检查通过率合理 |
| 2d | `src/filters.py` — quality 子命令 | 同上，跑 compile+quality 过滤 |
| 2e | `src/filters.py` — score-ppl 子命令 | 用 2b 的 20 条跑 PPL 打分，检查 ppl 字段非 NaN |
| 2f | `src/filters.py` — score-binary 子命令 | 同上，检查 score 字段有分布差异 |
| 2g | `src/filters.py` — filter-topk 子命令 | 对 2e/2f 的 scored 文件取 top 25%（5 条），检查输出行数 |
| 2h | `src/train.py` | 用 2b 的 20 条训练 **10 步**，检查 checkpoint 保存、loss 下降 |
| 2i | `src/evaluate_evalplus.py` | 用 base model 跑 HumanEval（greedy），对比已知 baseline 值 |
| 2j | `src/evaluate_lcb.py` | 用 base model 跑 LCB，确认输出格式正确 |

### Phase 3: 组合 + 端到端 Smoke Test

| 步骤 | 产出 | 验证方法 |
|------|------|---------|
| 3a | `src/run_experiment.sh` | 编写统一 SLURM loop 脚本 |
| 3b | Smoke test | 对 **1 个模型 × 1 个 filter** 跑 **2 轮**（生成 100 条 → 过滤 → 训练 50 步 → 评估），验证全流程串通、CSV/JSON 输出正确 |
| 3c | 扩展 smoke test | 对 **4 个模型** 各跑 1 轮 no_filter，确认所有模型都能走通 |

### Phase 4: 正式提交

| 步骤 | 说明 |
|------|------|
| 4a | 分批提交 20 组 SLURM job（20 轮 × 3000 步） |
| 4b | 优先级：no_filter × 4 → compile × 4 → quality × 4 → ppl × 4 → binary × 4 |
| 4c | 预估资源：每组 1-3 天（1 GPU H200），全部 ~20-60 GPU-days |

### Phase 5: 结果汇总与分析

| 步骤 | 产出 |
|------|------|
| 5a | `src/aggregate_results.py` — 合并 20 组 CSV + 对比图 |
| 5b | 分析研究问题（规模效应、架构效应、filter 通用性、self-referential 评分退化） |

---

## 附录：V1 实验结果参考

V1 在 SantaCoder 上的最终结果（详见 `old/progress.md`）：

| 实验 | HumanEval (best) | MBPP (best) | 结论 |
|------|-------------------|-------------|------|
| Baseline | 0.189 | 0.352 | 原始模型 |
| No Filter (30k步) | 0.061 | 0.032 | 完全 collapse |
| Compile Filter (30k步) | 0.067 | 0.108 | MBPP 退化最慢 |
| Quality Filter (60k步) | 0.061 | 0.061 | 延缓但仍 collapse |
| PPL Filter (24k步) | 0.055 | 0.098 | 延缓但仍 collapse |
| Binary Filter (30k步) | 0.079(R1) | 0.175(R1) | R1 最高峰值，之后快速下降 |

**V1 关键发现：**
- 所有 filter 均延缓但不能阻止 collapse
- Compile filter 对 MBPP 保护效果最好
- Binary classifier 的自评分随 collapse 同步失真（pass rate 85%→94.4%，score median 1.01→3.19）
- Self-referential scoring failure 是重要发现，V2 将验证其在其他模型上的普遍性