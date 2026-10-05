import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

from src.config import load_model_config, load_experiment_config
from src.filters import ExecutionVerifier, content_hash, file_hash, mixture_counts, load_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
GATES = {"none", "compile", "quality", "compile+quality", "execution", "ppl", "binary"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_command(python, module, **options):
    command = [python, "-m", module]
    for key, value in options.items():
        if value is not None:
            command.extend(["--" + key, str(value)])
    return command


def output_state(paths):
    state = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing stage output: {path}")
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        files = [item for item in files if item.is_file()]
        if not files:
            raise ValueError(f"Empty stage output: {path}")
        for item in files:
            stat = item.stat()
            state[str(item)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return state


def stage(directory, name, command, outputs, resume):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / (name + ".done.json")
    signature = fingerprint(command)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if resume and saved["command_sha256"] == signature and saved["outputs"] == output_state(outputs):
            print(f"Completed stage: {directory.name}/{name}", flush=True)
            return
        raise ValueError(f"Stage metadata or outputs changed: {marker}")
    print(shlex.join(command), flush=True)
    with (directory / (name + ".log")).open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(marker, {"command_sha256": signature, "command": command, "outputs": output_state(outputs)})


def expand_runs(config, args):
    suite_names = list(config["suites"]) if args.suite == ["all"] else args.suite
    runs = []
    for suite_name in suite_names:
        if suite_name not in config["suites"]:
            raise ValueError(f"Unknown suite: {suite_name}")
        suite = config["suites"][suite_name]
        cases = suite.get("cases", [{"name": gate, "gate": gate} for gate in suite.get("gates", [])])
        for model in suite["models"]:
            if args.models and model not in args.models:
                continue
            for case in cases:
                if args.gates and case["gate"] not in args.gates:
                    continue
                spec = copy.deepcopy(config["defaults"])
                spec.update(config["models"][model])
                spec.update({key: value for key, value in suite.items() if key not in ("models", "cases", "gates")})
                spec.update(case)
                spec.update(suite=suite_name, model=model)
                spec.setdefault("noise", 0.0)
                spec.setdefault("anchor_fraction", 0.0)
                if args.rounds is not None:
                    spec["rounds"] = args.rounds
                    spec.pop("evaluation_rounds", None)
                if args.nl_fraction is not None:
                    spec["nl_fraction"] = args.nl_fraction
                if args.execution_image:
                    spec["execution"]["image"] = args.execution_image
                for seed in args.seeds if args.seeds is not None else spec["seeds"]:
                    runs.append(copy.deepcopy(dict(spec, seed=seed)))
    if not runs:
        raise ValueError("No experiments match the selected suites, models, and gates")
    identities = [(run["suite"], run["model"], run["name"], run["seed"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Experiment selections contain duplicates")
    for run in runs:
        if any(Path(run[key]).name != run[key] or run[key] in (".", "..") for key in ("suite", "model", "name")):
            raise ValueError("Experiment names must be single path components")
        if run["gate"] not in GATES or run["rounds"] < 1 or run["seed"] < 0:
            raise ValueError("Invalid gate, rounds, or seed")
        if not 0 <= run["noise"] <= 1 or (run["noise"] and run["gate"] in ("none", "ppl", "binary")):
            raise ValueError("Decision noise requires an exogenous gate")
        mixture_counts(run["accepted_samples"], run["anchor_fraction"], run["nl_fraction"])
        if run["gate"] in ("ppl", "binary"):
            if run["anchor_fraction"] or int(run["raw_samples"] * run["retained_fraction"]) != run["accepted_samples"]:
                raise ValueError("AI gates require a fixed raw pool and exact retained fraction")
        evaluation_rounds = run.get("evaluation_rounds", list(range(run["rounds"] + 1)))
        if 0 not in evaluation_rounds or run["rounds"] not in evaluation_rounds:
            raise ValueError("Evaluation must include the measured baseline and final round")
        if any(round_id < 0 or round_id > run["rounds"] for round_id in evaluation_rounds):
            raise ValueError("Evaluation checkpoint outside the training horizon")
        run["evaluation_rounds"] = evaluation_rounds
    return runs


def preflight(runs, args):
    prompt_rows = load_jsonl(args.prompt_pool)
    if not prompt_rows:
        raise ValueError("Prompt pool is empty")
    sources = {"prompt_pool": {"path": args.prompt_pool, "sha256": file_hash(args.prompt_pool)}}
    if any(run["nl_fraction"] for run in runs):
        if not args.natural_language:
            raise ValueError("These model recipes require --natural_language with code-related text")
        nl_rows = load_jsonl(args.natural_language)
        for run in runs:
            needed = mixture_counts(run["accepted_samples"], run["anchor_fraction"], run["nl_fraction"])[2]
            for round_id in range(1, run["rounds"] + 1):
                if sum(row.get("round", round_id) == round_id for row in nl_rows) < needed:
                    raise ValueError(f"Insufficient natural-language data for round {round_id}")
        sources["natural_language"] = {"path": args.natural_language, "sha256": file_hash(args.natural_language)}
    if any(run["anchor_fraction"] for run in runs):
        if not args.anchor:
            raise ValueError("The anchor ablation requires --anchor with externally verified rows")
        anchors = load_jsonl(args.anchor)
        if any(row.get("verified") is not True for row in anchors):
            raise ValueError("Every external anchor row must contain verified: true")
        anchor_hashes = {content_hash(row) for row in anchors}
        if len(anchor_hashes) != len(anchors):
            raise ValueError("Fresh anchors must have distinct contents")
        if anchor_hashes & {content_hash(row) for row in prompt_rows if "content" in row}:
            raise ValueError("The anchor set overlaps the prompt pool")
        for run in runs:
            needed = round(run["accepted_samples"] * run["anchor_fraction"]) * run["rounds"]
            if len(anchors) < needed:
                raise ValueError(f"Fresh anchors require at least {needed} verified rows")
        sources["anchor"] = {"path": args.anchor, "sha256": file_hash(args.anchor)}
    sources["code"] = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / "src").glob("*.py"))}
    images = {run["execution"]["image"] for run in runs}
    resolved = {image: ExecutionVerifier(image).image for image in images}
    for run in runs:
        run["execution"]["image"] = resolved[run["execution"]["image"]]
    sources["recipe_sha256"] = file_hash(args.config or ROOT / "src/config.py")
    for run in runs:
        sources[run["model"] + "_config_sha256"] = file_hash(ROOT / run["config"])
    return sources


def evaluate_checkpoint(spec, args, python, checkpoint, directory):
    evaluation = spec["evaluation"]
    metrics = {}
    for dataset in evaluation["datasets"]:
        options = {
            "config": str(ROOT / spec["config"]), "model_path": checkpoint,
            "temperature": evaluation["temperature"], "top_p": evaluation["top_p"],
            "n_samples": evaluation["n_samples"], "max_new_tokens": evaluation["max_new_tokens"],
            "seed": spec["seed"],
        }
        if dataset == "livecodebench":
            destination = directory / "livecodebench"
            command = make_command(python, "src.evaluate_lcb", **options, output_dir=destination,
                                   release_version=evaluation["release_version"],
                                   num_process_evaluate=evaluation["workers"])
            stage(directory, dataset, command, [destination], args.resume)
            score_files = list(destination.glob("*_scores.json"))
            if len(score_files) != 1:
                raise ValueError(f"Expected one LiveCodeBench score file in {destination}")
            scores = json.loads(score_files[0].read_text())
            names = ["livecodebench_pass1"]
        elif dataset in ("humaneval", "mbpp"):
            destination = directory / (dataset + ".jsonl")
            scores_file = directory / (dataset + "_scores.json")
            command = make_command(python, "src.evaluate_evalplus", **options,
                                   dataset=dataset, output_file=destination)
            stage(directory, dataset, command, [destination, scores_file], args.resume)
            scores = json.loads(scores_file.read_text())
            names = [dataset + "_pass1", dataset + "_plus_pass1"]
        else:
            raise ValueError(f"Unknown evaluation dataset: {dataset}")
        for name in names:
            value = scores.get(name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Missing or invalid {name} score")
            metrics[name] = value
    return metrics


def run_experiment(spec, args, sources, python):
    directory = Path(args.output_root) / spec["suite"] / spec["model"] / spec["name"] / f"seed_{spec['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_keys = ["prompt_pool", "code", "recipe_sha256", spec["model"] + "_config_sha256"]
        if spec["nl_fraction"]:
            source_keys.append("natural_language")
        if spec["anchor_fraction"]:
            source_keys.append("anchor")
        manifest = {"spec": spec, "sources": {key: sources[key] for key in source_keys}, "python": python}
        manifest_file = directory / "experiment.json"
        if manifest_file.exists():
            if not args.resume or json.loads(manifest_file.read_text()) != manifest:
                raise ValueError(f"Existing experiment requires matching configuration and --resume: {directory}")
        else:
            write_json(manifest_file, manifest)
        model_config = str(ROOT / spec["config"])
        checkpoint = load_model_config(model_config)["model_id"]
        baseline_dir = directory / "round_000"
        baseline = evaluate_checkpoint(spec, args, python, checkpoint, baseline_dir)
        write_json(baseline_dir / "metrics.json", {"round": 0, "checkpoint": checkpoint, "scores": baseline})
        for round_id in range(1, spec["rounds"] + 1):
            current = directory / f"round_{round_id:03d}"
            current.mkdir(parents=True, exist_ok=True)
            round_seed = spec["seed"] + round_id * 100003
            accepted = current / "accepted.jsonl"
            generation_report = current / "generation.json"
            selection_report = current / "selection.json"
            raw_sample = current / "raw_sample.jsonl"
            ai_gate = spec["gate"] in ("ppl", "binary")
            anchor_count = round(spec["accepted_samples"] * spec["anchor_fraction"])
            synthetic_count = spec["accepted_samples"] - anchor_count
            generated_file = current / "raw.jsonl" if ai_gate else accepted
            generated_count = spec["raw_samples"] if ai_gate else synthetic_count
            execution = spec["execution"]
            command = make_command(
                python, "src.generate", config=model_config, model_path=checkpoint,
                prompt_pool=args.prompt_pool, output_file=generated_file,
                report=generation_report, raw_sample=raw_sample, num_samples=generated_count,
                gate="none" if ai_gate else spec["gate"], noise=spec["noise"],
                candidate_batch=spec["candidate_batch"], max_candidates=spec["max_candidates"],
                prompt_tokens=spec["prompt_tokens"], max_new_tokens=spec["max_new_tokens"],
                batch_size=spec["generation_batch_size"], temperature=spec["temperature"], top_p=spec["top_p"],
                seed=round_seed, prompt_seed=spec["prompt_seed"],
                repetition_threshold=spec["repetition_threshold"], min_completion_tokens=spec["min_completion_tokens"],
                diagnostic_samples=execution["samples"], execution_image=execution["image"],
                execution_timeout=execution["timeout"], execution_workers=execution["workers"],
            )
            stage(current, "generate", command, [generated_file, generation_report, raw_sample], args.resume)
            if ai_gate:
                scored = current / "scored.jsonl"
                command = make_command(python, "src.filters", config=model_config,
                                       model_path=checkpoint, input_file=generated_file,
                                       output_file=scored, batch_size=spec["score_batch_size"])
                command.insert(3, "score-" + spec["gate"])
                stage(current, "score", command, [scored], args.resume)
                command = make_command(python, "src.filters", input=scored, output=accepted,
                                       report=selection_report, count=synthetic_count,
                                       fraction=spec["retained_fraction"], gate=spec["gate"])
                command.insert(3, "rank")
                stage(current, "select", command, [accepted, selection_report], args.resume)
            else:
                write_json(selection_report, json.loads(generation_report.read_text()))
            training_data = current / "training.jsonl"
            mixture_report = current / "mixture.json"
            command = make_command(
                python, "src.filters", input=accepted, output=training_data, report=mixture_report,
                count=spec["accepted_samples"], round=round_id, seed=round_seed, anchor_seed=spec["seed"],
                anchor=args.anchor if anchor_count else None, anchor_fraction=spec["anchor_fraction"],
                natural_language=args.natural_language if spec["nl_fraction"] else None, nl_fraction=spec["nl_fraction"],
            )
            command.insert(3, "mix")
            stage(current, "mix", command, [training_data, mixture_report], args.resume)
            diagnostics = {}
            for population, data_file in [("raw", raw_sample), ("accepted", accepted), ("training", training_data)]:
                report = current / (population + "_execution.json")
                command = make_command(python, "src.filters", input=data_file, report=report,
                                       count=execution["samples"], timeout=execution["timeout"],
                                       image=execution["image"], workers=execution["workers"], seed=round_seed)
                command.insert(3, "execution")
                stage(current, population + "_execution", command, [report], args.resume)
                diagnostics[population] = json.loads(report.read_text())
            training = dict(spec["training"])
            training["warmup_steps"] = training["warmup_steps"] if round_id == 1 else 0
            train_dir = current / "model"
            next_checkpoint = train_dir / "final_checkpoint"
            training_report = current / "training_metrics.json"
            command = make_command(
                python, "src.train", config=model_config, model_path=checkpoint,
                local_data_path=training_data, output_dir=train_dir, metrics_file=training_report, seed=round_seed,
                batch_size=spec["batch_size"], gradient_accumulation_steps=spec["gradient_accumulation_steps"], **training,
            )
            stage(current, "train", command, [next_checkpoint, training_report], args.resume)
            checkpoint = str(next_checkpoint)
            scores = evaluate_checkpoint(spec, args, python, checkpoint, current) if round_id in spec["evaluation_rounds"] else {}
            write_json(current / "metrics.json", {
                "round": round_id, "checkpoint": checkpoint, "scores": scores,
                "generation": json.loads(generation_report.read_text()),
                "selection": json.loads(selection_report.read_text()),
                "mixture": json.loads(mixture_report.read_text()), "execution": diagnostics,
                "training": json.loads(training_report.read_text()),
            })
        write_json(directory / "complete.json", {"rounds": spec["rounds"], "manifest_sha256": fingerprint(manifest)})


def experiments_main(argv=None):
    parser = argparse.ArgumentParser(description="Plan paper experiments; --execute launches the selected runs")
    parser.add_argument("--config", help="Optional experiment YAML; defaults are embedded in src/config.py")
    parser.add_argument("--suite", nargs="+", default=["main"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--gates", nargs="+", choices=sorted(GATES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--prompt_pool", required=True)
    parser.add_argument("--natural_language", help="JSONL with content and optional round")
    parser.add_argument("--anchor", help="JSONL with content and verified: true; fresh rows for every round")
    parser.add_argument("--nl_fraction", type=float)
    parser.add_argument("--execution_image")
    parser.add_argument("--output_root", default=str(ROOT / "results/paper"))
    parser.add_argument("--python_by_model", action="append", default=[], metavar="MODEL=PYTHON")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for field in ("config", "prompt_pool", "natural_language", "anchor", "output_root"):
        value = getattr(args, field)
        if value:
            setattr(args, field, str(Path(value).expanduser().resolve()))
    config = load_experiment_config(args.config)
    if args.models and set(args.models) - set(config["models"]):
        parser.error("Unknown model selection")
    runs = expand_runs(config, args)
    interpreters = {}
    for item in args.python_by_model:
        model, separator, python = item.partition("=")
        if not separator or model not in config["models"] or not python:
            parser.error("--python_by_model must be MODEL=PYTHON for a configured model")
        interpreters[model] = str(Path(python).expanduser().resolve())
    if not args.execute:
        print(json.dumps({"experiments": runs, "inputs": {
            "prompt_pool": args.prompt_pool, "natural_language": args.natural_language,
            "anchor": args.anchor, "output_root": args.output_root,
        }}, indent=2))
        return
    sources = preflight(runs, args)
    for spec in runs:
        run_experiment(spec, args, sources, interpreters.get(spec["model"], sys.executable))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "experiments":
    experiments_main(sys.argv[2:])
    raise SystemExit(0)


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

from src.fim import get_fim_token_ids, permute


def get_args():
    parser = argparse.ArgumentParser(description="Self-play fine-tuning (V2)")

    parser.add_argument("--config", type=str, required=True,
                        help="Model config YAML (e.g. configs/santacoder.yaml)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model ID or local checkpoint (default: config model_id)")

    parser.add_argument("--local_data_path", type=str, required=True,
                        help="Training data JSONL file")
    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fim_rate", type=float, default=0.5)
    parser.add_argument("--fim_spm_rate", type=float, default=0.5)

    parser.add_argument("--num_of_sequences", type=int, default=1024)
    parser.add_argument("--size_valid_set", type=int, default=0,
                        help="Validation set size (0 = no validation)")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics_file", type=str)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--skip_final_save", action="store_true",
                        help="Debug only: run training without writing final checkpoint")

    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples,
                           desc="Estimating chars/token ratio"):
        text = example[data_column]
        total_characters += len(text)
        total_tokens += len(tokenizer(text).tokens())
    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):

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


def run_training(args):
    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]
    set_seed(args.seed)

    print(f"Model: {cfg['short_name']} ({model_path})")
    print(f"Data:  {args.local_data_path}")
    print(f"Steps: {args.max_steps}, LR: {args.learning_rate}, "
          f"BS: {args.batch_size} x {args.gradient_accumulation_steps}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg.get("trust_remote_code", False),
        use_cache=not args.gradient_checkpointing,
    )

    train_dataset, valid_dataset = create_datasets(tokenizer, cfg, args)

    import inspect

    eval_kwargs = {}
    if valid_dataset is not None:
        parameters = inspect.signature(TrainingArguments.__init__).parameters
        eval_key = (
            "eval_strategy"
            if "eval_strategy" in parameters
            else "evaluation_strategy"
        )
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

    train_result = trainer.train()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Peak CUDA memory allocated: {peak_mem:.2f} GiB")
        print(f"Peak CUDA memory reserved: {peak_reserved:.2f} GiB")

    final_ckpt = os.path.join(args.output_dir, "final_checkpoint")
    if args.skip_final_save:
        print("Skipping final checkpoint save (--skip_final_save).")
    else:
        trainer.save_model(final_ckpt)
        tokenizer.save_pretrained(final_ckpt)

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

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    final_loss = train_losses[-1] if train_losses else None
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
        with open(args.metrics_file, "w") as stream:
            json.dump({**train_result.metrics, "final_logged_loss": final_loss,
                       "global_step": trainer.state.global_step, "seed": args.seed}, stream, indent=2)
    print("\nTraining complete.")
    print(f"Final loss: {final_loss}")
    if not args.skip_final_save:
        print(f"Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    args = get_args()
    run_training(args)
