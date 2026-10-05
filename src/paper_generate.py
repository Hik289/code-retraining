import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.config import load_model_config
from src.execution import ExecutionVerifier
from src.paper_data import file_hash, read_jsonl, write_json, write_jsonl


def static_gate(sample, gate, repetition_threshold, min_tokens):
    if gate in ("compile", "compile+quality", "execution"):
        try:
            compile(sample["content"], "<candidate>", "exec")
        except (SyntaxError, ValueError, OverflowError):
            return False, "compile"
    if gate in ("quality", "compile+quality"):
        lines = [line.strip() for line in sample["content"].splitlines() if line.strip()]
        repetitions = sum(count - 1 for count in Counter(lines).values())
        if lines and repetitions / len(lines) > repetition_threshold:
            return False, "repetition"
        if len(sample["completion_token_ids"]) < min_tokens:
            return False, "length"
    return True, None


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-pool recursive generation with auditable gate accounting")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_pool", required=True, help="JSONL rows with prompt or content; optional test_code")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raw_sample", required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--candidate_batch", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_candidates", type=int, default=200000)
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_seed", type=int, default=0)
    parser.add_argument("--gate", choices=["none", "compile", "quality", "compile+quality", "execution"], default="none")
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--repetition_threshold", type=float, default=0.3)
    parser.add_argument("--min_completion_tokens", type=int, default=50)
    parser.add_argument("--diagnostic_samples", type=int, default=500)
    parser.add_argument("--execution_image", default="python:3.11-slim")
    parser.add_argument("--execution_timeout", type=float, default=5)
    parser.add_argument("--execution_workers", type=int, default=4)
    args = parser.parse_args()
    for field in ("num_samples", "candidate_batch", "batch_size", "prompt_tokens", "max_new_tokens", "diagnostic_samples", "execution_workers"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.max_candidates < args.num_samples:
        parser.error("--max_candidates must cover --num_samples")
    if not 0 <= args.noise <= 1 or (args.noise and args.gate == "none"):
        parser.error("Noise requires a gate and a probability in [0, 1]")
    if not 0 <= args.repetition_threshold <= 1 or args.min_completion_tokens < 0:
        parser.error("Invalid quality thresholds")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Sampling requires positive temperature and top_p in (0, 1]")
    return args


def main():
    args = parse_args()
    pool = read_jsonl(args.prompt_pool)
    if not pool:
        raise ValueError("The prompt pool is empty")
    for row in pool:
        if not isinstance(row.get("prompt", row.get("content")), str):
            raise ValueError("Each prompt row must contain prompt or content text")
    random.Random(args.prompt_seed).shuffle(pool)
    verifier = ExecutionVerifier(args.execution_image, args.execution_timeout) if args.gate == "execution" else None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    cfg = load_model_config(args.config)
    if args.prompt_tokens + args.max_new_tokens > cfg["max_context"]:
        raise ValueError("Prompt and completion budgets exceed the model context")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=cfg["trust_remote_code"], torch_dtype=torch.bfloat16,
    ).cuda().eval()
    prompts = []
    for row in pool:
        ids = tokenizer.encode(row.get("prompt", row.get("content")), add_special_tokens=False)[:args.prompt_tokens]
        if not ids:
            raise ValueError("The prompt pool contains an empty token sequence")
        prompts.append((row, ids))
    noise_rng = random.Random(args.seed + 1000003)
    reservoir_rng = random.Random(args.seed + 2000003)
    generated = clean_passed = gate_passed = flipped = 0
    reject_counts = Counter()
    accepted = []
    reservoir = []
    with ThreadPoolExecutor(max_workers=args.execution_workers) as executor:
        while len(accepted) < args.num_samples and generated < args.max_candidates:
            wave_size = min(args.candidate_batch, args.max_candidates - generated)
            if args.gate == "none":
                wave_size = min(wave_size, args.num_samples - len(accepted))
            candidates = []
            for start in range(0, wave_size, args.batch_size):
                records = [prompts[(generated + index) % len(prompts)] for index in range(start, min(start + args.batch_size, wave_size))]
                features = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for _, ids in records]
                inputs = tokenizer.pad(features, padding=True, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=args.temperature, top_p=args.top_p, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for (row, prompt_ids), output in zip(records, outputs):
                    completion_ids = output[inputs["input_ids"].shape[1]:].tolist()
                    terminal_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
                    completion_ids = completion_ids[:next((i for i, token in enumerate(completion_ids) if token in terminal_ids), len(completion_ids))]
                    sample = {
                        "content": tokenizer.decode(prompt_ids + completion_ids, skip_special_tokens=True),
                        "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                        "prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids,
                        "prompt_tokens": len(prompt_ids),
                        "candidate_id": generated + len(candidates),
                        "prompt_id": row.get("id", (generated + len(candidates)) % len(prompts)),
                    }
                    if row.get("test_code"):
                        sample["test_code"] = row["test_code"]
                    candidates.append(sample)
            decisions = [static_gate(row, args.gate, args.repetition_threshold, args.min_completion_tokens) for row in candidates]
            if verifier:
                positions = [i for i, (ok, _) in enumerate(decisions) if ok]
                statuses = executor.map(verifier, (candidates[i] for i in positions))
                for index, status in zip(positions, statuses):
                    decisions[index] = (status == "ok", None if status == "ok" else status)
                    candidates[index]["execution_status"] = status
            for sample, (clean_ok, reason) in zip(candidates, decisions):
                generated += 1
                clean_passed += int(clean_ok)
                flip = noise_rng.random() < args.noise
                flipped += int(flip)
                ok = clean_ok != flip
                gate_passed += int(ok)
                sample["clean_gate_accepted"] = clean_ok
                sample["gate_accepted"] = ok
                sample["gate_flipped"] = flip
                if ok and len(accepted) < args.num_samples:
                    accepted.append(sample)
                if not ok:
                    reject_counts["noise" if flip else reason or "gate"] += 1
                if len(reservoir) < args.diagnostic_samples:
                    reservoir.append(sample)
                else:
                    index = reservoir_rng.randrange(generated)
                    if index < len(reservoir):
                        reservoir[index] = sample
            print(json.dumps({"generated": generated, "gate_passed": gate_passed, "selected": len(accepted)}), flush=True)
    complete = len(accepted) == args.num_samples
    report = {
        "complete": complete, "model_path": args.model_path, "gate": args.gate,
        "seed": args.seed, "prompt_seed": args.prompt_seed,
        "prompt_pool_sha256": file_hash(args.prompt_pool), "prompt_pool_size": len(pool),
        "generated": generated, "clean_gate_passed": clean_passed,
        "gate_passed": gate_passed, "selected": len(accepted),
        "clean_gate_pass_rate": clean_passed / generated,
        "raw_gate_pass_rate": gate_passed / generated,
        "acceptance_mass": gate_passed / generated,
        "retained_fraction": len(accepted) / generated,
        "candidates_per_5000_accepted": generated * 5000 / len(accepted) if accepted else None,
        "expected_candidates_per_5000": generated * 5000 / gate_passed if gate_passed else None,
        "noise_probability": args.noise, "effective_flip_rate": flipped / generated,
        "flipped": flipped, "rejections": dict(reject_counts),
        "accepted_overflow": gate_passed - len(accepted),
        "execution_image": verifier.image if verifier else None,
    }
    write_jsonl(args.raw_sample, reservoir)
    write_json(args.report, report)
    if not complete:
        raise RuntimeError(f"Candidate budget exhausted: accepted {len(accepted)} of {args.num_samples}")
    write_jsonl(args.output_file, accepted)


if __name__ == "__main__":
    main()
