import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def check_compile(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except (SyntaxError, ValueError, OverflowError):
        return False


def check_repetition(code: str, threshold: float = 0.5) -> bool:
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rate = repeated / len(lines)
    return rate <= threshold


def check_length(code: str, tokenizer, prompt_tokens: int,
                 min_completion_tokens: int = 50) -> bool:
    tokens = tokenizer(code, truncation=False)["input_ids"]
    completion_len = len(tokens) - prompt_tokens
    return completion_len >= min_completion_tokens


def cmd_compile(args):
    samples = load_jsonl(args.input_file)
    kept = [s for s in samples if check_compile(s["content"])]
    write_jsonl(kept, args.output_file)

    rate = len(kept) / len(samples) * 100 if samples else 0
    print("\n===== Compile Filter =====")
    print(f"Input:  {len(samples)}")
    print(f"Passed: {len(kept)} ({rate:.1f}%)")
    print(f"Output: {args.output_file}")


def cmd_quality(args):
    from transformers import AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path or cfg["model_id"],
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    samples = load_jsonl(args.input_file)
    kept = []
    reject_counts = Counter()

    for s in samples:
        text = s["content"]
        if args.require_compile and not check_compile(text):
            reject_counts["compile"] += 1
            continue
        if not check_repetition(text, threshold=args.repetition_threshold):
            reject_counts["repetition"] += 1
            continue
        if not check_length(text, tokenizer, s.get("prompt_tokens", args.prompt_tokens),
                            min_completion_tokens=args.min_completion_tokens):
            reject_counts["length"] += 1
            continue
        kept.append(s)

    write_jsonl(kept, args.output_file)

    rate = len(kept) / len(samples) * 100 if samples else 0
    print("\n===== Quality Filter =====")
    print(f"Input:  {len(samples)}")
    print(f"Passed: {len(kept)} ({rate:.1f}%)")
    for reason, cnt in sorted(reject_counts.items()):
        print(f"  Rejected [{reason}]: {cnt}")
    print(f"Output: {args.output_file}")


def compute_ppl_batch(texts, model, tokenizer, prompt_tokens, max_length, device, token_sequences=None):
    import torch

    if token_sequences is None:
        encodings = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
    else:
        sequences = [ids[:max_length] for ids in token_sequences]
        encodings = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in sequences],
            padding=True, return_tensors="pt",
        ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i in range(labels.size(0)):
        real_start = (attention_mask[i] == 0).sum().item()
        prefix_length = prompt_tokens[i] if isinstance(prompt_tokens, list) else prompt_tokens
        mask_end = min(real_start + prefix_length, labels.size(1))
        labels[i, :mask_end] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    mask = shift_labels != -100
    ppls = []
    for i in range(input_ids.size(0)):
        sample_mask = mask[i]
        if sample_mask.sum() == 0:
            ppls.append(float("inf"))
        else:
            mean_loss = per_token_loss[i][sample_mask].mean().item()
            ppls.append(math.exp(mean_loss) if mean_loss < 709 else float("inf"))

    return ppls


def cmd_score_ppl(args):
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_ppls = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring PPL"):
        batch = samples[i:i + args.batch_size]
        texts = [s["content"] for s in batch]
        prefixes = [s.get("prompt_tokens", args.prompt_tokens) for s in batch]
        token_sequences = None
        if all("prompt_token_ids" in s and "completion_token_ids" in s for s in batch):
            token_sequences = [s["prompt_token_ids"] + s["completion_token_ids"] for s in batch]
            prefixes = [len(s["prompt_token_ids"]) for s in batch]
        ppls = compute_ppl_batch(texts, model, tokenizer,
                                 prefixes, args.max_length, device, token_sequences)
        all_ppls.extend(ppls)

    for sample, ppl in zip(samples, all_ppls):
        sample["ppl"] = ppl
    write_jsonl(samples, args.output_file)

    finite_ppls = [p for p in all_ppls if math.isfinite(p)]
    if finite_ppls:
        arr = np.array(finite_ppls)
        print("\n===== PPL Stats =====")
        print(f"Samples: {len(all_ppls)} (finite: {len(finite_ppls)}, "
              f"inf: {len(all_ppls) - len(finite_ppls)})")
        print(f"min={arr.min():.2f}  p25={np.percentile(arr, 25):.2f}  "
              f"median={np.median(arr):.2f}  p75={np.percentile(arr, 75):.2f}  "
              f"max={arr.max():.2f}")
    print(f"Output: {args.output_file}")


BINARY_TEMPLATE = "\n# quality: "


def score_binary_batch(texts, model, tokenizer, template_ids, good_id, bad_id,
                       max_content_tokens, device):
    import torch

    seqs = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[-max_content_tokens:]
        ids = ids + template_ids
        seqs.append(ids)

    max_len = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id

    input_ids_list = []
    attention_masks = []
    for s in seqs:
        pad_len = max_len - len(s)
        input_ids_list.append([pad_id] * pad_len + s)
        attention_masks.append([0] * pad_len + [1] * len(s))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    last_logits = outputs.logits[:, -1, :]
    scores = (last_logits[:, good_id] - last_logits[:, bad_id]).tolist()
    return scores


def cmd_score_binary(args):
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import load_model_config

    cfg = load_model_config(args.config)
    model_path = args.model_path or cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()
    device = next(model.parameters()).device

    good_token = cfg.get("binary_good_token", " good")
    bad_token = cfg.get("binary_bad_token", " bad")
    good_ids = tokenizer.encode(good_token, add_special_tokens=False)
    bad_ids = tokenizer.encode(bad_token, add_special_tokens=False)
    good_id = good_ids[0]
    bad_id = bad_ids[0]
    print(f"Good token: {repr(good_token)} => IDs {good_ids}, using {good_id}")
    print(f"Bad token:  {repr(bad_token)} => IDs {bad_ids}, using {bad_id}")
    if len(good_ids) > 1 or len(bad_ids) > 1:
        print("WARNING: multi-token target; only using first subword logit")

    template_ids = tokenizer.encode(BINARY_TEMPLATE, add_special_tokens=False)
    print(f"Template: {repr(BINARY_TEMPLATE)} => {len(template_ids)} tokens")

    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    all_scores = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring binary"):
        batch = samples[i:i + args.batch_size]
        texts = [s.get("completion", s["content"]) for s in batch]
        scores = score_binary_batch(
            texts, model, tokenizer, template_ids,
            good_id, bad_id, args.max_content_tokens, device,
        )
        all_scores.extend(scores)

    for sample, score in zip(samples, all_scores):
        sample["score"] = score
    write_jsonl(samples, args.output_file)

    finite_scores = [s for s in all_scores if math.isfinite(s)]
    if finite_scores:
        arr = np.array(finite_scores)
        print("\n===== Binary Score Stats =====")
        print(f"Samples: {len(all_scores)} (finite: {len(finite_scores)}, "
              f"nan/inf: {len(all_scores) - len(finite_scores)})")
        print(f"min={arr.min():.4f}  median={np.median(arr):.4f}  "
              f"max={arr.max():.4f}")
        print(f"good (score>0): {(arr > 0).sum()} ({100*(arr > 0).mean():.1f}%)")
    print(f"Output: {args.output_file}")


def cmd_filter_topk(args):
    import numpy as np
    if not 0 < args.top_percent <= 100:
        raise ValueError("--top_percent must be in (0, 100]")

    samples = load_jsonl(args.input_file)
    field = args.score_field

    valid = [(i, s) for i, s in enumerate(samples)
             if math.isfinite(s.get(field, float("inf")))]
    invalid_count = len(samples) - len(valid)

    valid.sort(key=lambda x: x[1][field], reverse=(not args.ascending))

    keep_count = max(1, int(len(samples) * args.top_percent / 100))
    if len(valid) < keep_count:
        raise ValueError(f"Only {len(valid)} finite scores for {keep_count} required samples")
    kept = valid[:keep_count]

    out_samples = []
    for _, s in kept:
        out = {k: v for k, v in s.items() if k != field}
        out_samples.append(out)
    write_jsonl(out_samples, args.output_file)

    if valid:
        all_vals = np.array([s[field] for _, s in valid])
        kept_vals = np.array([s[field] for _, s in kept])
        order = "ascending (lower=better)" if args.ascending else "descending (higher=better)"
        print(f"\n===== Top-K Filter ({field}, {order}) =====")
        print(f"Input:   {len(samples)} (valid: {len(valid)}, invalid: {invalid_count})")
        print(f"Keeping: {keep_count} (top {args.top_percent}%)")
        print(f"All {field}:  min={all_vals.min():.4f}  median={np.median(all_vals):.4f}  "
              f"max={all_vals.max():.4f}")
        print(f"Kept {field}: min={kept_vals.min():.4f}  median={np.median(kept_vals):.4f}  "
              f"max={kept_vals.max():.4f}")
    print(f"Output: {args.output_file}")


EXECUTION_RUNNER = """
import json
import os
import signal
import sys

payload = json.load(sys.stdin)
exit_process = os._exit
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
scope = {'__name__': '__main__'}
signal.signal(signal.SIGALRM, lambda *_: exit_process(29))
signal.setitimer(signal.ITIMER_REAL, payload['timeout'])
try:
    exec(compile(payload['content'], '<candidate>', 'exec'), scope)
    if payload.get('test_code'):
        exec(compile(payload['test_code'], '<tests>', 'exec'), scope)
except ModuleNotFoundError:
    exit_process(21)
except ImportError:
    exit_process(20)
except (SyntaxError, IndentationError):
    exit_process(22)
except NameError:
    exit_process(23)
except AttributeError:
    exit_process(24)
except TypeError:
    exit_process(25)
except ValueError:
    exit_process(26)
except AssertionError:
    exit_process(28)
except BaseException:
    exit_process(27)
exit_process(0)
"""

EXECUTION_STATUSES = {
    0: "ok", 20: "ImportError", 21: "ModuleNotFoundError",
    22: "SyntaxError", 23: "NameError", 24: "AttributeError",
    25: "TypeError", 26: "ValueError", 27: "other_error",
    28: "AssertionError", 29: "timeout", 137: "resource_limit",
}


class ExecutionVerifier:
    def __init__(self, image="python:3.11-slim", timeout=5):
        if timeout <= 0:
            raise ValueError("Execution timeout must be positive")
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True, capture_output=True, text=True, timeout=30,
        )
        self.image = result.stdout.strip()
        if not self.image.startswith("sha256:"):
            raise RuntimeError("Cannot resolve the local execution image")
        self.timeout = timeout

    def __call__(self, sample):
        name = "retraining-" + uuid.uuid4().hex
        command = [
            "docker", "run", "--rm", "--pull", "never", "--name", name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "-i", self.image,
            "python", "-I", "-c", EXECUTION_RUNNER,
        ]
        try:
            result = subprocess.run(
                command, input=json.dumps(dict(sample, timeout=self.timeout)), text=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=self.timeout + 30,
            )
            if result.returncode in (125, 126, 127):
                raise RuntimeError(f"Execution container failed: {result.stderr.strip()}")
            return EXECUTION_STATUSES.get(result.returncode, "other_error")
        except subprocess.TimeoutExpired:
            return "container_timeout"
        finally:
            cleanup = subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=15,
            )
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                raise RuntimeError(f"Execution cleanup failed: {cleanup.stderr.strip()}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(row):
    return hashlib.sha256(row["content"].encode("utf-8")).hexdigest()


def select_ranked(args):
    rows = load_jsonl(args.input)
    target = int(len(rows) * args.fraction)
    if target != args.count:
        raise ValueError(f"Retained fraction gives {target} samples, expected {args.count}")
    field = "ppl" if args.gate == "ppl" else "score"
    valid = [row for row in rows if math.isfinite(row.get(field, float("nan")))]
    if len(valid) < target:
        raise ValueError(f"Only {len(valid)} finite scores for {target} required samples")
    ranked = sorted(valid, key=lambda row: row[field], reverse=args.gate == "binary")
    selected = ranked[:target]
    write_jsonl(selected, args.output)
    values = sorted(row[field] for row in valid)
    write_json(args.report, {
        "gate": args.gate,
        "generated": len(rows), "gate_passed": target, "selected": target,
        "raw_gate_pass_rate": target / len(rows),
        "retained_fraction": target / len(rows),
        "candidates_per_5000_accepted": len(rows) * 5000 / target,
        "expected_candidates_per_5000": len(rows) * 5000 / target,
        "invalid_scores": len(rows) - len(valid),
        "score_min": values[0], "score_median": values[len(values) // 2],
        "score_max": values[-1], "selection_threshold": selected[-1][field],
        "binary_positive_score_fraction": (
            sum(row[field] > 0 for row in valid) / len(valid) if args.gate == "binary" else None
        ),
        "input_sha256": file_hash(args.input),
    })


def mixture_counts(total, anchor_fraction, nl_fraction):
    if total <= 0 or not 0 <= anchor_fraction < 1 or not 0 <= nl_fraction < 1:
        raise ValueError("Invalid mixture size or fraction")
    anchor_count = round(total * anchor_fraction)
    nl_count = round(total * nl_fraction)
    code_count = total - anchor_count - nl_count
    if code_count <= 0:
        raise ValueError("The training mixture must contain synthetic code")
    return code_count, anchor_count, nl_count


def mix_training_data(args):
    code_count, anchor_count, nl_count = mixture_counts(args.count, args.anchor_fraction, args.nl_fraction)
    synthetic = load_jsonl(args.input)
    rng = random.Random(args.seed)
    if len(synthetic) < code_count:
        raise ValueError("Not enough selected synthetic samples")
    rows = [dict(row, training_source="synthetic") for row in rng.sample(synthetic, code_count)]
    sources = {}
    if anchor_count:
        if not args.anchor:
            raise ValueError("An external verified anchor JSONL is required")
        anchors = load_jsonl(args.anchor)
        if any(row.get("verified") is not True for row in anchors):
            raise ValueError("Every anchor row must contain verified: true")
        if len({content_hash(row) for row in anchors}) != len(anchors):
            raise ValueError("Anchor contents must be unique across rounds")
        order = list(range(len(anchors)))
        random.Random(args.anchor_seed).shuffle(order)
        start = (args.round - 1) * anchor_count
        chosen = [anchors[index] for index in order[start:start + anchor_count]]
        if len(chosen) != anchor_count:
            raise ValueError("Not enough fresh anchors for this round")
        occupied = {content_hash(row) for row in synthetic}
        if any(content_hash(row) in occupied for row in chosen):
            raise ValueError("Anchor content overlaps the synthetic set")
        rows.extend(dict(row, training_source="verified_anchor") for row in chosen)
        sources["anchor_sha256"] = file_hash(args.anchor)
        sources["anchor_content_sha256"] = [content_hash(row) for row in chosen]
    if nl_count:
        if not args.natural_language:
            raise ValueError("A code-related natural-language JSONL is required")
        natural_language = load_jsonl(args.natural_language)
        eligible = [row for row in natural_language if row.get("round", args.round) == args.round]
        if len(eligible) < nl_count:
            raise ValueError("Not enough natural-language samples for this round")
        chosen = rng.sample(eligible, nl_count)
        rows.extend(dict(row, training_source="natural_language") for row in chosen)
        sources["natural_language_sha256"] = file_hash(args.natural_language)
    rng.shuffle(rows)
    if any(not isinstance(row.get("content"), str) or not row["content"].strip() for row in rows):
        raise ValueError("Training rows must contain nonempty content strings")
    if any(row.get("test_code") is not None and not isinstance(row["test_code"], str) for row in rows):
        raise ValueError("test_code must be a string when supplied")
    training_rows = [{"content": row["content"], "training_source": row["training_source"],
                      "source_sha256": content_hash(row), "test_code": row.get("test_code") or ""} for row in rows]
    write_jsonl(training_rows, args.output)
    write_json(args.report, {
        "round": args.round, "seed": args.seed, "count": len(rows),
        "synthetic_selected": len(synthetic), "synthetic_used": code_count,
        "anchor_count": anchor_count, "natural_language_count": nl_count,
        "anchor_fraction": anchor_count / len(rows),
        "natural_language_fraction": nl_count / len(rows),
        "synthetic_sha256": file_hash(args.input), **sources,
    })


def analyze_execution(args):
    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError("Cannot analyze an empty candidate set")
    indices = random.Random(args.seed).sample(range(len(rows)), min(args.count, len(rows)))
    sampled = [rows[index] for index in indices]
    verifier = ExecutionVerifier(args.image, args.timeout)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        statuses = list(executor.map(verifier, sampled))
    counts = Counter(statuses)
    compiled = 0
    for row in sampled:
        try:
            compile(row["content"], "<candidate>", "exec")
            compiled += 1
        except (SyntaxError, ValueError, OverflowError):
            pass
    relaxed = counts["ok"] + counts["ImportError"] + counts["ModuleNotFoundError"]
    write_json(args.report, {
        "input_sha256": file_hash(args.input), "population": len(rows),
        "evaluated": len(sampled), "seed": args.seed, "timeout_seconds": args.timeout,
        "image": verifier.image, "counts": dict(counts),
        "compile_rate": compiled / len(sampled),
        "execution_strict_rate": counts["ok"] / len(sampled),
        "execution_relaxed_rate": relaxed / len(sampled),
        "samples_with_test_code": sum(bool(row.get("test_code")) for row in sampled),
        "sample_indices": indices, "statuses": statuses,
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Self-play data filtering (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compile", help="Keep only samples that pass compile()")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)

    p = sub.add_parser("quality", help="Repetition and length checks, optionally with compilation")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local path (default: config model_id)")
    p.add_argument("--prompt_tokens", type=int, default=1024)
    p.add_argument("--require_compile", action="store_true")
    p.add_argument("--repetition_threshold", type=float, default=0.3)
    p.add_argument("--min_completion_tokens", type=int, default=50)

    p = sub.add_parser("score-ppl", help="Score samples by PPL (needs GPU)")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local checkpoint (default: config model_id)")
    p.add_argument("--prompt_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)

    p = sub.add_parser("score-binary", help="Score samples by binary classifier (needs GPU)")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--config", required=True, help="Model config YAML")
    p.add_argument("--model_path", default=None,
                   help="HF model ID or local checkpoint (default: config model_id)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_content_tokens", type=int, default=2000)

    p = sub.add_parser("filter-topk", help="Keep top K%% by score field")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--score_field", required=True,
                   help="JSON field to sort by (e.g. 'ppl' or 'score')")
    p.add_argument("--top_percent", type=float, default=25)
    p.add_argument("--ascending", action="store_true",
                   help="Sort ascending (lower=better, e.g. PPL). "
                        "Default: descending (higher=better, e.g. binary score)")

    ranked = sub.add_parser("rank")
    ranked.add_argument("--gate", required=True, choices=["ppl", "binary"])
    ranked.add_argument("--fraction", type=float, default=0.25)
    mixed = sub.add_parser("mix")
    mixed.add_argument("--anchor")
    mixed.add_argument("--anchor_fraction", type=float, default=0)
    mixed.add_argument("--anchor_seed", type=int, default=0)
    mixed.add_argument("--natural_language")
    mixed.add_argument("--nl_fraction", type=float, default=0)
    mixed.add_argument("--round", type=int, required=True)
    mixed.add_argument("--seed", type=int, default=0)
    execution = sub.add_parser("execution")
    execution.add_argument("--image", default="python:3.11-slim")
    execution.add_argument("--timeout", type=float, default=5)
    execution.add_argument("--workers", type=int, default=4)
    execution.add_argument("--seed", type=int, default=0)
    for command in [ranked, mixed, execution]:
        command.add_argument("--input", required=True)
        command.add_argument("--report", required=True)
        command.add_argument("--count", type=int, default=500 if command is execution else 5000)
    for command in [ranked, mixed]:
        command.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command in {"rank", "mix", "execution"} and args.count <= 0:
        parser.error("--count must be positive")
    if args.command == "rank" and not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.command == "mix" and args.round < 1:
        parser.error("--round must be positive")
    if args.command == "execution" and args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main():
    args = parse_args()

    dispatch = {
        "compile": cmd_compile,
        "quality": cmd_quality,
        "score-ppl": cmd_score_ppl,
        "score-binary": cmd_score_binary,
        "filter-topk": cmd_filter_topk,
        "rank": select_ranked,
        "mix": mix_training_data,
        "execution": analyze_execution,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
