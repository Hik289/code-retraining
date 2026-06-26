"""Probe stable self-play generation batch sizes for different models.

This runs a few real generation batches using the same prompt/max_new_tokens
settings as formal self-play and reports whether each batch size succeeds.
"""
import argparse
import gc
import glob
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_model_config


# Path to a locally cached The Stack (dedup, Python) Arrow dataset directory.
# Set the THE_STACK_ARROW_CACHE environment variable to point at it, or pass
# --local_dataset_path on the command line. Falls back to streaming from the Hub.
DEFAULT_ARROW_CACHE = os.environ.get("THE_STACK_ARROW_CACHE", "")


def parse_args():
    p = argparse.ArgumentParser(description="Probe generation batch size")
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--min-file-tokens", type=int, default=512)
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shuffle-buffer", type=int, default=50000)
    p.add_argument("--local-dataset-path", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_dataset_iter(args):
    arrow_dir = args.local_dataset_path or DEFAULT_ARROW_CACHE
    data_files = sorted(
        f
        for f in glob.glob(os.path.join(arrow_dir, "*.arrow"))
        if not os.path.basename(f).startswith("cache-")
    )
    if not data_files:
        raise FileNotFoundError(f"No .arrow files found in {arrow_dir}")
    dataset = load_dataset("arrow", data_files=data_files, split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    return iter(dataset)


def collect_prompts(data_iter, tokenizer, batch_size, prompt_tokens, min_file_tokens):
    prompts = []
    while len(prompts) < batch_size:
        example = next(data_iter)
        text = example["content"]
        token_ids = tokenizer(text, truncation=False)["input_ids"]
        if len(token_ids) < min_file_tokens:
            continue
        prompt_ids = token_ids[:prompt_tokens]
        prompts.append(tokenizer.decode(prompt_ids, skip_special_tokens=True))
    return prompts


def cleanup(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = parse_args()
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    data_iter = load_dataset_iter(args)
    results = []

    for batch_size in args.batch_sizes:
        print(f"\n=== Probing batch_size={batch_size} ===", flush=True)
        status = {
            "batch_size": batch_size,
            "ok": False,
            "num_batches": args.num_batches,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
        }
        model = None
        try:
            cleanup()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=cfg.get("trust_remote_code", False),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to("cuda:0")
            model.eval()

            torch.cuda.reset_peak_memory_stats()
            batch_times = []

            for batch_idx in range(args.num_batches):
                prompts = collect_prompts(
                    data_iter,
                    tokenizer,
                    batch_size,
                    args.prompt_tokens,
                    args.min_file_tokens,
                )
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.prompt_tokens,
                ).to("cuda:0")

                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                elapsed = time.time() - start
                batch_times.append(elapsed)
                print(
                    f"batch {batch_idx + 1}/{args.num_batches} ok, "
                    f"shape={tuple(outputs.shape)}, time={elapsed:.2f}s",
                    flush=True,
                )

            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            status.update(
                {
                    "ok": True,
                    "avg_batch_time_sec": round(sum(batch_times) / len(batch_times), 2),
                    "peak_mem_gib": round(peak_gib, 2),
                    "peak_reserved_gib": round(reserved_gib, 2),
                }
            )
            print(
                f"SUCCESS batch_size={batch_size} "
                f"peak_mem={peak_gib:.2f}GiB reserved={reserved_gib:.2f}GiB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            status["error"] = f"OOM: {exc}"
            print(f"OOM at batch_size={batch_size}", flush=True)
        except StopIteration:
            status["error"] = "Dataset exhausted while collecting prompts"
            print("Dataset exhausted unexpectedly", flush=True)
        except Exception as exc:  # noqa: BLE001
            status["error"] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL at batch_size={batch_size}: {exc}", flush=True)
        finally:
            results.append(status)
            cleanup(model)

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
