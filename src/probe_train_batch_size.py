"""Probe stable per-device training batch sizes with real backward passes.

This script uses synthetic token batches with the formal seq_length and runs a
few optimizer steps with gradient checkpointing enabled. It is meant to find a
stable per-device training batch size ceiling for each model.
"""
import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_model_config


def parse_args():
    p = argparse.ArgumentParser(description="Probe training batch size")
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()


def cleanup(model=None, optimizer=None):
    if optimizer is not None:
        del optimizer
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_batch(tokenizer, batch_size, seq_length, vocab_size, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    input_ids = torch.randint(
        low=0,
        high=max(vocab_size - 1, 1),
        size=(batch_size, seq_length),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, -1] = pad_id
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for batch_size in args.batch_sizes:
        print(f"\n=== Probing train batch_size={batch_size} ===", flush=True)
        status = {
            "batch_size": batch_size,
            "ok": False,
            "steps": args.steps,
            "seq_length": args.seq_length,
        }
        model = None
        optimizer = None
        try:
            cleanup()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=cfg.get("trust_remote_code", False),
                torch_dtype=torch.bfloat16,
                use_cache=False,
                local_files_only=True,
            ).to(device)
            model.train()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            vocab_size = int(getattr(model.config, "vocab_size", None) or len(tokenizer))
            torch.cuda.reset_peak_memory_stats()

            step_times = []
            losses = []
            for step in range(args.steps):
                input_ids, attention_mask, labels = make_batch(
                    tokenizer, batch_size, args.seq_length, vocab_size, device
                )
                optimizer.zero_grad(set_to_none=True)
                start = time.time()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.time() - start
                step_times.append(elapsed)
                losses.append(float(loss.detach().cpu()))
                print(
                    f"step {step + 1}/{args.steps} ok, "
                    f"loss={losses[-1]:.4f}, time={elapsed:.2f}s",
                    flush=True,
                )

            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            status.update(
                {
                    "ok": True,
                    "avg_step_time_sec": round(sum(step_times) / len(step_times), 2),
                    "peak_mem_gib": round(peak_gib, 2),
                    "peak_reserved_gib": round(reserved_gib, 2),
                    "last_loss": round(losses[-1], 4),
                }
            )
            print(
                f"SUCCESS train batch_size={batch_size} "
                f"peak_mem={peak_gib:.2f}GiB reserved={reserved_gib:.2f}GiB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            status["error"] = f"OOM: {exc}"
            print(f"OOM at train batch_size={batch_size}", flush=True)
        except Exception as exc:  # noqa: BLE001
            status["error"] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL at train batch_size={batch_size}: {exc}", flush=True)
        finally:
            results.append(status)
            cleanup(model, optimizer)

    payload = {
        "model": cfg["short_name"],
        "model_path": model_path,
        "results": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved probe results to {args.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
