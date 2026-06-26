"""scripts/lcb_generate.py — 为 LiveCodeBench 生成代码补全

绕开 LiveCodeBench 的 vLLM runner（与 transformers==4.35.2 不兼容），
自己加载模型生成代码，输出 LiveCodeBench evaluate 兼容的 JSON。

用法：
    python scripts/lcb_generate.py \
        --model_path bigcode/santacoder \
        --output_dir livecodebench_results \
        --release_version release_v1

评估：
    python scripts/lcb_generate.py \
        --model_path bigcode/santacoder \
        --output_dir livecodebench_results \
        --release_version release_v1 \
        --evaluate_only
"""
import argparse
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./livecodebench_results")
    parser.add_argument("--release_version", type=str, default="release_v1")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per problem")
    parser.add_argument("--debug", action="store_true",
                        help="Only run on first 15 problems")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--timeout", type=int, default=6,
                        help="Timeout per test case in seconds")
    parser.add_argument("--num_process_evaluate", type=int, default=16)
    return parser.parse_args()


# LiveCodeBench 问题的 few-shot examples（从 LCB 仓库提取）
# 用 1-shot prompt，与 GenericBase 风格一致
LCB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "LiveCodeBench"
)

STOP_SEQUENCES = [
    "\n### Question", "\n### Answer", "\n---",
    "<|endoftext|>", "<file_sep>",
]


def load_few_shot_examples():
    """加载 LiveCodeBench 的 few-shot examples。"""
    func_path = os.path.join(
        LCB_DIR, "lcb_runner/prompts/few_shot_examples/generation/func.json"
    )
    stdin_path = os.path.join(
        LCB_DIR, "lcb_runner/prompts/few_shot_examples/generation/stdin.json"
    )
    with open(func_path) as f:
        func_examples = json.load(f)
    with open(stdin_path) as f:
        stdin_examples = json.load(f)
    return func_examples, stdin_examples


def format_prompt(question, func_examples, stdin_examples):
    """构造 GenericBase 风格的 1-shot prompt。

    与 LiveCodeBench 的 get_base_model_question_template_answer 一致。
    """
    has_starter = bool(question.starter_code)
    examples = func_examples if has_starter else stdin_examples
    example = examples[0]

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
    """截断到第一个 stop sequence。"""
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 LiveCodeBench 数据（直接从 HF jsonl 加载，绕过 datasets loading script 兼容问题）
    sys.path.insert(0, LCB_DIR)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    VERSION_TO_FILE = {
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl",
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl",
    }
    jsonl_name = VERSION_TO_FILE.get(args.release_version)
    if jsonl_name is None:
        raise ValueError(f"Unknown release version: {args.release_version}")

    print(f"Loading LiveCodeBench ({args.release_version})...")
    jsonl_path = hf_hub_download(
        "livecodebench/code_generation_lite", jsonl_name, repo_type="dataset"
    )
    with open(jsonl_path) as f:
        benchmark = [CodeGenerationProblem(**json.loads(line)) for line in f]
    print(f"Loaded {len(benchmark)} problems")
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    if args.debug:
        benchmark = benchmark[:15]
        print(f"Debug mode: using {len(benchmark)} problems")

    # 输出路径
    model_id = args.model_path.replace("/", "--")
    output_path = os.path.join(
        args.output_dir,
        f"{model_id}_{args.release_version}.json"
    )
    eval_path = output_path.replace(".json", "_eval.json")
    eval_all_path = output_path.replace(".json", "_eval_all.json")

    if not args.evaluate_only:
        # 加载 few-shot examples
        func_examples, stdin_examples = load_few_shot_examples()

        # 加载模型
        print(f"Loading model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        model.eval()
        device = torch.device("cuda:0")

        greedy = args.temperature == 0.0 or args.n_samples == 1

        # 生成
        save_results = []
        for question in tqdm(benchmark, desc="Generating"):
            prompt = format_prompt(question, func_examples, stdin_examples)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # 检查是否超过模型 context window（SantaCoder: 2048）
            max_ctx = getattr(model.config, "n_positions", 2048)
            if input_ids.shape[-1] + args.max_new_tokens > max_ctx:
                # 截断 prompt，保留末尾（包含实际问题）
                max_prompt_len = max_ctx - args.max_new_tokens
                if input_ids.shape[-1] > max_prompt_len:
                    input_ids = input_ids[:, -max_prompt_len:]

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
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
                    output = model.generate(input_ids, use_cache=True, **gen_kwargs)

                generated_ids = output[0, input_ids.size(-1):]
                completion = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                completion = truncate_at_stop(completion, STOP_SEQUENCES)
                completion = completion.replace("\t", "    ")

                output_list.append(completion)
                # GenericBase extraction: just strip
                code_list.append(completion.strip())

            save_results.append(
                question.insert_output(output_list, code_list)
            )

        # 保存生成结果
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=4)
        print(f"Saved {len(save_results)} results to {output_path}")

    else:
        # evaluate_only: 加载已有结果
        with open(output_path, "r") as f:
            save_results = json.load(f)
        print(f"Loaded {len(save_results)} results from {output_path}")

    # 评估
    print("Running evaluation...")
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import extract_instance_results

    eval_samples = [q.get_evaluation_sample() for q in benchmark]
    generations = [r["code_list"] for r in save_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    print(f"\n===== LiveCodeBench Results ({args.release_version}) =====")
    print(f"pass@1: {metrics[0]['pass@1']:.4f}")
    for key, val in metrics[0].items():
        if key != "detail":
            print(f"  {key}: {val}")

    # 保存评估结果
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=4)

    graded = extract_instance_results(metrics[1])
    save_eval_results = [
        question.insert_output_evaluation(
            r["output_list"], r["code_list"], graded_list, metadata=meta
        )
        for question, r, graded_list, meta in zip(
            benchmark, save_results, graded, metrics[2]
        )
    ]
    with open(eval_all_path, "w") as f:
        json.dump(save_eval_results, f, indent=4)

    print(f"Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
