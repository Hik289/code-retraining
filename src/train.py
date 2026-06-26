"""train.py — Multi-model self-play fine-tuning

Config-driven training script supporting 4 models with different transformers versions.
Training data is local JSONL from self-play generation (--local_data_path).

Usage:
    python src/train.py \
        --config configs/santacoder.yaml \
        --local_data_path results/santacoder/compile_filter/generated_data/round1.jsonl \
        --max_steps 3000 \
        --output_dir results/santacoder/compile_filter/round1
"""
import argparse
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

from src.config import load_model_config
from src.fim import get_fim_token_ids, permute


# ============================================================
# Argument parsing
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="Self-play fine-tuning (V2)")

    # Model config
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML (e.g. configs/santacoder.yaml)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model ID or local checkpoint (default: config model_id)")

    # Data
    parser.add_argument("--local_data_path", type=str, required=True,
                        help="Training data JSONL file")
    parser.add_argument("--data_column", type=str, default="content")

    # Training hyperparams
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    # FIM
    parser.add_argument("--fim_rate", type=float, default=0.5)
    parser.add_argument("--fim_spm_rate", type=float, default=0.5)

    # Data loading
    parser.add_argument("--num_of_sequences", type=int, default=1024)
    parser.add_argument("--size_valid_set", type=int, default=0,
                        help="Validation set size (0 = no validation)")

    # Output and logging
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--skip_final_save", action="store_true",
                        help="Debug only: run training without writing final checkpoint")

    # Gradient checkpointing (on by default)
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================
def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """Sample-based estimation of chars-per-token ratio."""
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples,
                           desc="Estimating chars/token ratio"):
        text = example[data_column]
        total_characters += len(text)
        total_tokens += len(tokenizer(text).tokens())
    return total_characters / total_tokens


# ============================================================
# ConstantLengthDataset
# ============================================================
class ConstantLengthDataset(IterableDataset):
    """Packs variable-length texts into fixed-length token sequences with FIM augmentation.

    Flow: fill char buffer -> batch tokenize -> optional FIM permutation ->
          concatenate with EOS -> slice into seq_length chunks -> shuffle.
    labels = input_ids (CLM next-token prediction).
    """

    def __init__(self, tokenizer, dataset, config, infinite=False,
                 seq_length=2048, num_of_sequences=1024,
                 chars_per_token=3.6, content_field="content",
                 fim_rate=0.5, fim_spm_rate=0.5, seed=0):
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

        # FIM token IDs (config-driven, supports all 4 models)
        self.suffix_tok_id = None
        if fim_rate > 0:
            (self.suffix_tok_id, self.prefix_tok_id,
             self.middle_tok_id, self.pad_tok_id) = get_fim_token_ids(tokenizer, config)
            if self.suffix_tok_id is None:
                print("WARNING: FIM tokens not found in tokenizer, disabling FIM")
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
                    tokenized_input, np_rng = permute(
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
# Dataset creation
# ============================================================
def create_datasets(tokenizer, config, args):
    dataset = load_dataset("json",
                           data_files={"train": args.local_data_path},
                           split="train")

    if args.size_valid_set > 0:
        split = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed)
        train_data = split["train"]
        valid_data = split["test"]
    else:
        train_data = dataset
        valid_data = None

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"Chars/token ratio: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, config, infinite=True,
        seq_length=args.seq_length, num_of_sequences=args.num_of_sequences,
        chars_per_token=chars_per_token, content_field=args.data_column,
        fim_rate=args.fim_rate, fim_spm_rate=args.fim_spm_rate, seed=args.seed,
    )

    valid_dataset = None
    if valid_data is not None:
        valid_dataset = ConstantLengthDataset(
            tokenizer, valid_data, config, infinite=False,
            seq_length=args.seq_length, num_of_sequences=args.num_of_sequences,
            chars_per_token=chars_per_token, content_field=args.data_column,
            fim_rate=0, seed=args.seed,
        )

    return train_dataset, valid_dataset


# ============================================================
# Training
# ============================================================
def run_training(args):
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]
    set_seed(args.seed)

    print(f"Model: {cfg['short_name']} ({model_path})")
    print(f"Data:  {args.local_data_path}")
    print(f"Steps: {args.max_steps}, LR: {args.learning_rate}, "
          f"BS: {args.batch_size} x {args.gradient_accumulation_steps}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        use_cache=not args.gradient_checkpointing,
    )

    # ---- Dataset ----
    train_dataset, valid_dataset = create_datasets(tokenizer, cfg, args)

    # ---- TrainingArguments ----
    # transformers<4.36 uses evaluation_strategy, >=4.36 uses eval_strategy
    import transformers
    tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])

    eval_kwargs = {}
    if valid_dataset is not None:
        eval_key = "eval_strategy" if tf_version >= (4, 36) else "evaluation_strategy"
        eval_kwargs[eval_key] = "steps"
        eval_kwargs["eval_steps"] = args.eval_freq

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
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        dataloader_drop_last=True,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
        **eval_kwargs,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- Train ----
    trainer.train()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Peak CUDA memory allocated: {peak_mem:.2f} GiB")
        print(f"Peak CUDA memory reserved: {peak_reserved:.2f} GiB")

    # ---- Save final checkpoint ----
    final_ckpt = os.path.join(args.output_dir, "final_checkpoint")
    if args.skip_final_save:
        print("Skipping final checkpoint save (--skip_final_save).")
    else:
        trainer.save_model(final_ckpt)
        tokenizer.save_pretrained(final_ckpt)

        # Training uses use_cache=False (gradient checkpointing), restore for inference
        import json
        gen_cfg_path = os.path.join(final_ckpt, "generation_config.json")
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as f:
                gen_cfg = json.load(f)
            gen_cfg["use_cache"] = True
            with open(gen_cfg_path, "w") as f:
                json.dump(gen_cfg, f, indent=2)
        else:
            with open(gen_cfg_path, "w") as f:
                json.dump({"use_cache": True}, f, indent=2)

    # ---- Report final loss ----
    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    final_loss = train_losses[-1] if train_losses else None
    print(f"\nTraining complete.")
    print(f"Final loss: {final_loss}")
    if not args.skip_final_save:
        print(f"Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    args = get_args()
    run_training(args)
