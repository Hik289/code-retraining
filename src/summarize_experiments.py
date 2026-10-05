import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from src.paper_data import write_json


METRICS = ["humaneval_pass1", "humaneval_plus_pass1", "mbpp_pass1", "mbpp_plus_pass1", "livecodebench_pass1"]


def valid_score(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1


def trajectory_statistics(records, metric, horizon):
    measured = {row["round"]: row.get("scores", {}).get(metric) for row in records}
    baseline, endpoint = measured.get(0), measured.get(horizon)
    result = {"baseline": baseline, "endpoint": endpoint, "retention": None,
              "retention_area": None, "endpoint_normalized_slope": None,
              "ols_normalized_slope": None, "measured_rounds": sorted(
                  round_id for round_id, value in measured.items() if round_id <= horizon and valid_score(value)
              )}
    if not valid_score(baseline) or baseline == 0:
        result["normalization_status"] = "missing_or_zero_baseline"
        return result
    if valid_score(endpoint):
        result["retention"] = endpoint / baseline
        result["endpoint_normalized_slope"] = (endpoint / baseline - 1) / horizon
    if any(not valid_score(measured.get(round_id)) for round_id in range(horizon + 1)):
        result["normalization_status"] = "incomplete_trajectory"
        return result
    normalized = [measured[round_id] / baseline for round_id in range(horizon + 1)]
    result["retention_area"] = sum((left + right) / 2 for left, right in zip(normalized, normalized[1:]))
    x_mean = horizon / 2
    y_mean = statistics.mean(normalized)
    denominator = sum((round_id - x_mean) ** 2 for round_id in range(horizon + 1))
    result["ols_normalized_slope"] = sum((round_id - x_mean) * (value - y_mean) for round_id, value in enumerate(normalized)) / denominator
    result["normalization_status"] = "complete"
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def summarize(root, horizon):
    summary = []
    per_round = []
    for manifest_path in sorted(root.rglob("experiment.json")):
        manifest = json.loads(manifest_path.read_text())
        spec = manifest["spec"]
        identity = {key: spec[key] for key in ("suite", "model", "name", "gate", "seed", "noise", "anchor_fraction", "nl_fraction")}
        records = [json.loads(path.read_text()) for path in sorted(manifest_path.parent.glob("round_*/metrics.json"))]
        for record in records:
            row = dict(identity, round=record["round"], **record.get("scores", {}))
            selection = record.get("selection", {})
            generation = record.get("generation", {})
            mixture = record.get("mixture", {})
            for field in ("generated", "gate_passed", "selected", "raw_gate_pass_rate", "retained_fraction", "candidates_per_5000_accepted", "expected_candidates_per_5000", "binary_positive_score_fraction", "score_median"):
                row[field] = selection.get(field)
            row["train_loss"] = record.get("training", {}).get("train_loss")
            row["steps_total"] = record["round"] * spec["training"]["max_steps"]
            row["effective_flip_rate"] = generation.get("effective_flip_rate")
            row["acceptance_mass"] = selection.get("raw_gate_pass_rate")
            for field in ("synthetic_used", "anchor_count", "natural_language_count"):
                row[field] = mixture.get(field)
            row["candidates_per_training_set"] = selection.get("generated")
            for population, execution in record.get("execution", {}).items():
                for field in ("compile_rate", "execution_strict_rate", "execution_relaxed_rate", "evaluated"):
                    row[population + "_" + field] = execution.get(field)
            per_round.append(row)
        endpoint = next((row for row in records if row["round"] == horizon), {})
        for metric in METRICS:
            stats = trajectory_statistics(records, metric, horizon)
            if not stats["measured_rounds"]:
                continue
            summary.append(dict(identity, metric=metric, horizon=horizon,
                                experiment_complete=(manifest_path.parent / "complete.json").exists(),
                                endpoint_gate_accounting=endpoint.get("selection", {}),
                                endpoint_execution=endpoint.get("execution", {}), **stats))
    return summary, per_round


def across_seeds(rows):
    keys = ("suite", "model", "name", "gate", "noise", "anchor_fraction", "nl_fraction", "metric", "horizon")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, members in grouped.items():
        output = dict(zip(keys, identity))
        seeds = [row["seed"] for row in members]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed results for {identity}")
        output["seeds"] = sorted(seeds)
        for field in ("endpoint", "retention", "retention_area", "endpoint_normalized_slope", "ols_normalized_slope"):
            values = [row[field] for row in members if row[field] is not None]
            output[field + "_n"] = len(values)
            output[field + "_mean"] = statistics.mean(values) if values else None
            output[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(output)
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize measured recursive-training results without fixed baseline constants")
    parser.add_argument("--root", type=Path, default=Path("results/paper"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    summary, per_round = summarize(args.root, args.horizon)
    if not summary:
        raise ValueError("No measured benchmark results were found")
    output = args.output or args.root / "summary"
    write_json(output / "summary.json", {"horizon": args.horizon, "experiments": summary, "across_seeds": across_seeds(summary)})
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_round.csv", per_round)
    write_csv(output / "across_seeds.csv", across_seeds(summary))


if __name__ == "__main__":
    main()
