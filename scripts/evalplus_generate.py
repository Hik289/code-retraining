import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"])
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy decoding")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples", type=int, default=1)
    return parser.parse_args()


STOP_SEQUENCES_HUMANEVAL = [
    "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```",
    "<file_sep>", "<|endoftext|>",
]
STOP_SEQUENCES_MBPP = [
    "\nclass", "\nassert", '\n"""', "\nprint", "\nif",
    "<file_sep>", "<|endoftext|>",
]


def truncate_at_stop(text, stop_sequences):
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        tasks = get_human_eval_plus()
    else:
        from evalplus.data import get_mbpp_plus
        tasks = get_mbpp_plus()

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

    greedy = args.temperature == 0.0

    results = []
    for task_id, task in tqdm(tasks.items(), desc=f"Generating {args.dataset}"):
        prompt = task["prompt"].strip()

        for sample_idx in range(args.n_samples):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

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

            with torch.no_grad():
                output = model.generate(input_ids, use_cache=True, **gen_kwargs)

            generated_ids = output[0, input_ids.size(-1):]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            stop_seqs = STOP_SEQUENCES_MBPP if args.dataset == "mbpp" else STOP_SEQUENCES_HUMANEVAL
            completion = truncate_at_stop(completion, stop_seqs)
            completion = completion.replace("\t", "    ")

            identifier = f"{task_id}_{sample_idx}"
            results.append({
                "task_id": task_id,
                "completion": completion,
                "_identifier": identifier,
            })

    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"完成：{len(results)} 条 samples 写入 {args.output_file}")


if __name__ == "__main__":
    main()
