"""evaluate_evalplus.py — Generate completions and evaluate with EvalPlus

Handles both HumanEval/HumanEval+ and MBPP/MBPP+.
Loads model ourselves (bypassing evalplus.codegen which uses attn_implementation
that SantaCoder doesn't support), generates completions, then calls
evalplus.evaluate to get pass@1 scores.

Usage:
    python src/evaluate_evalplus.py \
        --config configs/santacoder.yaml \
        --model_path results/santacoder/compile/round1/final_checkpoint \
        --dataset humaneval \
        --output_file evalplus_results/humaneval/santacoder_compile_r1.jsonl
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="EvalPlus evaluation (V2)")
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model ID or local checkpoint (default: config model_id)")
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"])
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL for generated samples")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per task")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run on first N tasks only (for smoke tests)")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Only generate samples and write a placeholder score file")
    return parser.parse_args()


def truncate_at_stop(text, stop_sequences):
    """Truncate text at the first occurrence of any stop sequence."""
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def generate_samples(args, cfg):
    """Generate completions for all tasks in the dataset."""
    model_path = args.model_path or cfg["model_id"]

    # Load tasks
    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        tasks = get_human_eval_plus()
        stop_key = "stop_sequences_humaneval"
    else:
        from evalplus.data import get_mbpp_plus
        tasks = get_mbpp_plus()
        stop_key = "stop_sequences_mbpp"

    if args.limit is not None:
        tasks = dict(list(tasks.items())[: args.limit])

    stop_sequences = cfg.get(stop_key, [])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()

    greedy = args.temperature == 0.0

    results = []
    for task_id, task in tqdm(tasks.items(), desc=f"Generating {args.dataset}"):
        # Strip trailing whitespace — critical for SantaCoder to avoid empty output
        prompt = task["prompt"].strip()

        for sample_idx in range(args.n_samples):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

            # Extract only the generated portion
            generated_ids = output[0, input_ids.size(-1):]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion = truncate_at_stop(completion, stop_sequences)
            # Normalize tabs to 4 spaces (evalplus convention)
            completion = completion.replace("\t", "    ")

            results.append({
                "task_id": task_id,
                "completion": completion,
                "_identifier": f"{task_id}_{sample_idx}",
            })

    return results


def run_evaluate(samples_file, dataset):
    """Run evalplus.evaluate and parse results.

    Returns dict with pass@1 scores for base and plus versions.
    """
    # Delete old eval_results.json to force re-evaluation
    # (evalplus.evaluate reuses a cached result file if present)
    result_path = samples_file.replace(".jsonl", "_eval_results.json")
    if os.path.isfile(result_path):
        os.remove(result_path)
        print(f"Removed old {result_path}")

    from evalplus.evaluate import evaluate
    evaluate(dataset=dataset, samples=samples_file, i_just_wanna_run=True)

    # Parse the eval_results.json written by evalplus
    if not os.path.isfile(result_path):
        print(f"WARNING: {result_path} not found after evaluation")
        return {}

    with open(result_path) as f:
        eval_results = json.load(f)

    # Extract pass@1 from the eval results
    # Structure: {"eval": {task_id: [{"base_status": "pass"|"fail", "plus_status": ...}, ...]}}
    scores = {}
    eval_data = eval_results.get("eval", {})
    if not eval_data:
        return scores

    n_tasks = len(eval_data)
    base_pass = 0
    plus_pass = 0

    for task_id, task_results in eval_data.items():
        if not task_results:
            continue
        # For n_samples=1, task_results has 1 entry per sample
        # Count task as passing if first sample passes (greedy)
        sample = task_results[0]
        if sample.get("base_status") == "pass":
            base_pass += 1
        if sample.get("plus_status") == "pass":
            plus_pass += 1

    scores[f"{dataset}_pass1"] = round(base_pass / n_tasks, 4) if n_tasks > 0 else 0.0
    scores[f"{dataset}_plus_pass1"] = round(plus_pass / n_tasks, 4) if n_tasks > 0 else 0.0

    return scores


def main():
    args = parse_args()
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    print(f"Model: {cfg['short_name']} ({model_path})")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}, n_samples: {args.n_samples}")

    # Generate completions
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results = generate_samples(args, cfg)

    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} samples to {args.output_file}")

    # Evaluate
    if args.skip_evaluate:
        print("\nSkipping EvalPlus evaluation (generation-only smoke mode).")
        scores = {
            f"{args.dataset}_pass1": None,
            f"{args.dataset}_plus_pass1": None,
            "limited_tasks": args.limit,
            "evaluation_skipped": True,
        }
    else:
        print(f"\nRunning EvalPlus evaluation...")
        scores = run_evaluate(args.output_file, args.dataset)
    for k, v in scores.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Write scores to a companion JSON
    scores_file = args.output_file.replace(".jsonl", "_scores.json")
    scores["model"] = cfg["short_name"]
    scores["model_path"] = model_path
    scores["dataset"] = args.dataset
    scores["temperature"] = args.temperature
    scores["n_samples"] = args.n_samples
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {scores_file}")


if __name__ == "__main__":
    main()
