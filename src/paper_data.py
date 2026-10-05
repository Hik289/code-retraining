import argparse
import hashlib
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.execution import ExecutionVerifier


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


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
    rows = read_jsonl(args.input)
    target = int(len(rows) * args.fraction)
    if target != args.count:
        raise ValueError(f"Retained fraction gives {target} samples, expected {args.count}")
    field = "ppl" if args.gate == "ppl" else "score"
    valid = [row for row in rows if math.isfinite(row.get(field, float("nan")))]
    if len(valid) < target:
        raise ValueError(f"Only {len(valid)} finite scores for {target} required samples")
    ranked = sorted(valid, key=lambda row: row[field], reverse=args.gate == "binary")
    selected = ranked[:target]
    write_jsonl(args.output, selected)
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
    synthetic = read_jsonl(args.input)
    rng = random.Random(args.seed)
    if len(synthetic) < code_count:
        raise ValueError("Not enough selected synthetic samples")
    rows = [dict(row, training_source="synthetic") for row in rng.sample(synthetic, code_count)]
    sources = {}
    if anchor_count:
        if not args.anchor:
            raise ValueError("An external verified anchor JSONL is required")
        anchors = read_jsonl(args.anchor)
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
        natural_language = read_jsonl(args.natural_language)
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
    write_jsonl(args.output, training_rows)
    write_json(args.report, {
        "round": args.round, "seed": args.seed, "count": len(rows),
        "synthetic_selected": len(synthetic), "synthetic_used": code_count,
        "anchor_count": anchor_count, "natural_language_count": nl_count,
        "anchor_fraction": anchor_count / len(rows),
        "natural_language_fraction": nl_count / len(rows),
        "synthetic_sha256": file_hash(args.input), **sources,
    })


def analyze_execution(args):
    rows = read_jsonl(args.input)
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


def main():
    parser = argparse.ArgumentParser(description="Selection, data mixtures, and execution diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    ranked = sub.add_parser("rank")
    ranked.add_argument("--gate", required=True, choices=["ppl", "binary"])
    ranked.add_argument("--fraction", type=float, default=0.25)
    ranked.set_defaults(function=select_ranked)
    mixed = sub.add_parser("mix")
    mixed.add_argument("--anchor")
    mixed.add_argument("--anchor_fraction", type=float, default=0)
    mixed.add_argument("--anchor_seed", type=int, default=0)
    mixed.add_argument("--natural_language")
    mixed.add_argument("--nl_fraction", type=float, default=0)
    mixed.add_argument("--round", type=int, required=True)
    mixed.add_argument("--seed", type=int, default=0)
    mixed.set_defaults(function=mix_training_data)
    execution = sub.add_parser("execution")
    execution.add_argument("--image", default="python:3.11-slim")
    execution.add_argument("--timeout", type=float, default=5)
    execution.add_argument("--workers", type=int, default=4)
    execution.add_argument("--seed", type=int, default=0)
    execution.set_defaults(function=analyze_execution)
    for command in [ranked, mixed, execution]:
        command.add_argument("--input", required=True)
        command.add_argument("--report", required=True)
        command.add_argument("--count", type=int, default=500 if command is execution else 5000)
    for command in [ranked, mixed]:
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.command == "rank" and not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.command == "mix" and args.round < 1:
        parser.error("--round must be positive")
    if args.command == "execution" and args.workers < 1:
        parser.error("--workers must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
