"""LiveCodeBench evaluation for V2 experiments.

Generates code for LiveCodeBench code_generation_lite and evaluates pass@1.
This intentionally bypasses the upstream vLLM runner so SantaCoder can keep its
transformers==4.35.2 environment.
"""
import argparse
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import load_model_config


LCB_DIR = os.path.join(PROJECT_DIR, "LiveCodeBench")
FEW_SHOT_DIR = os.path.join(
    LCB_DIR, "lcb_runner", "prompts", "few_shot_examples", "generation"
)

VERSION_TO_FILE = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
}

STOP_SEQUENCES = [
    "\n### Question",
    "\n### Answer",
    "\n---",
    "<|endoftext|>",
    "<file_sep>",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LiveCodeBench evaluation (V2)")
    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML or short name")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model ID or local checkpoint (default: config model_id)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for generated results and score JSON")
    parser.add_argument("--release_version", type=str, default="release_v1",
                        choices=sorted(VERSION_TO_FILE))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per problem")
    parser.add_argument("--timeout", type=int, default=6,
                        help="Timeout per test case in seconds")
    parser.add_argument("--num_process_evaluate", type=int, default=16)
    parser.add_argument("--debug", action="store_true",
                        help="Run on first 15 problems only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run on first N problems after sorting")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Skip generation and evaluate an existing result file")
    return parser.parse_args()


def load_few_shot_examples():
    with open(os.path.join(FEW_SHOT_DIR, "func.json")) as f:
        func_examples = json.load(f)
    with open(os.path.join(FEW_SHOT_DIR, "stdin.json")) as f:
        stdin_examples = json.load(f)
    return func_examples, stdin_examples


def format_prompt(question, func_examples, stdin_examples):
    """Build the same 1-shot GenericBase prompt as LiveCodeBench."""
    has_starter = bool(question.starter_code)
    example = (func_examples if has_starter else stdin_examples)[0]

    def make_example_block(q, sample_code, answer):
        block = "### Question\n"
        block += q + "\n\n"
        if has_starter:
            block += "### Starter Code\n"
            block += sample_code + "\n\n"
        block += "### Answer\n\n"
        block += answer
        if answer:
            block += "\n\n"
        return block

    prompt = make_example_block(
        example["question"], example.get("sample_code", ""), example["answer"]
    )
    prompt += make_example_block(
        question.question_content, question.starter_code, ""
    )
    return prompt


def truncate_at_stop(text, stop_sequences):
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def load_benchmark(release_version, debug=False, limit=None):
    sys.path.insert(0, LCB_DIR)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    jsonl_path = hf_hub_download(
        "livecodebench/code_generation_lite",
        VERSION_TO_FILE[release_version],
        repo_type="dataset",
    )
    with open(jsonl_path) as f:
        benchmark = [CodeGenerationProblem(**json.loads(line)) for line in f]

    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    if debug:
        benchmark = benchmark[:15]
    if limit is not None:
        benchmark = benchmark[:limit]
    return benchmark


def get_context_limit(model, cfg):
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return int(cfg.get("max_context", 2048))


def generate_results(args, cfg, benchmark, output_path):
    model_path = args.model_path or cfg["model_id"]
    func_examples, stdin_examples = load_few_shot_examples()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    greedy = args.temperature == 0.0
    max_context = get_context_limit(model, cfg)
    save_results = []

    for question in tqdm(benchmark, desc="Generating LiveCodeBench"):
        prompt = format_prompt(question, func_examples, stdin_examples)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

        if input_ids.shape[-1] + args.max_new_tokens > max_context:
            max_prompt_len = max_context - args.max_new_tokens
            if max_prompt_len <= 0:
                raise ValueError(
                    f"max_new_tokens={args.max_new_tokens} exceeds context {max_context}"
                )
            input_ids = input_ids[:, -max_prompt_len:]

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        output_list = []
        code_list = []
        for _ in range(args.n_samples):
            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
            generated_ids = output[0, input_ids.size(-1):]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion = truncate_at_stop(completion, STOP_SEQUENCES)
            completion = completion.replace("\t", "    ")
            output_list.append(completion)
            code_list.append(completion.strip())

        save_results.append(question.insert_output(output_list, code_list))

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    return save_results


def run_evaluate(args, benchmark, save_results):
    sys.path.insert(0, LCB_DIR)
    from lcb_runner.evaluation import codegen_metrics

    eval_samples = [q.get_evaluation_sample() for q in benchmark]
    generations = [r["code_list"] for r in save_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )
    return metrics


def main():
    args = parse_args()
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]
    os.makedirs(args.output_dir, exist_ok=True)

    model_id = model_path.replace("/", "--")
    suffix = "_debug" if args.debug or args.limit else ""
    output_path = os.path.join(
        args.output_dir,
        f"{cfg['short_name']}_{model_id}_{args.release_version}{suffix}.json",
    )
    eval_path = output_path.replace(".json", "_eval.json")
    scores_path = output_path.replace(".json", "_scores.json")

    print(f"Model: {cfg['short_name']} ({model_path})")
    print(f"LiveCodeBench: {args.release_version}")
    print(f"Temperature: {args.temperature}, n_samples: {args.n_samples}")

    benchmark = load_benchmark(args.release_version, args.debug, args.limit)
    print(f"Loaded {len(benchmark)} problems")

    if args.evaluate_only:
        with open(output_path) as f:
            save_results = json.load(f)
        print(f"Loaded existing generations from {output_path}")
    else:
        save_results = generate_results(args, cfg, benchmark, output_path)
        print(f"Wrote {len(save_results)} generations to {output_path}")

    print("\nRunning LiveCodeBench evaluation...")
    metrics = run_evaluate(args, benchmark, save_results)
    pass1 = float(metrics[0].get("pass@1", 0.0))
    print(f"  livecodebench_pass1: {pass1:.4f}")

    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    scores = {
        "livecodebench_pass1": round(pass1, 4),
        "model": cfg["short_name"],
        "model_path": model_path,
        "release_version": args.release_version,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "num_problems": len(benchmark),
    }
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
