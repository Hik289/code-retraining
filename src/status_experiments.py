"""Summarize V2 experiment progress from results CSVs and SLURM queue.

Usage:
    python src/status_experiments.py
    python src/status_experiments.py --root results_smoke
"""
import argparse
import csv
import os
import subprocess


MODELS = ["santacoder", "starcoder2", "qwen25", "codellama"]
FILTERS = ["none", "compile", "quality", "ppl", "binary"]


def read_last_row(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def queue_lines():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", "youruser"], text=True, stderr=subprocess.STDOUT
        )
    except Exception as exc:
        return [f"squeue unavailable: {exc}"]
    return out.strip().splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    args = parser.parse_args()

    print(f"Experiment status root: {args.root}")
    print("")
    header = (
        "model", "filter", "round", "steps", "HE", "HE+", "MBPP", "MBPP+",
        "LCB", "generated", "kept", "timestamp",
    )
    print(",".join(header))

    for model in MODELS:
        for filter_name in FILTERS:
            csv_path = os.path.join(args.root, model, filter_name, "results.csv")
            row = read_last_row(csv_path)
            if row is None:
                values = [model, filter_name, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
            else:
                values = [
                    model,
                    filter_name,
                    row.get("round", ""),
                    row.get("steps_total", ""),
                    row.get("humaneval_pass1", ""),
                    row.get("humaneval_plus_pass1", ""),
                    row.get("mbpp_pass1", ""),
                    row.get("mbpp_plus_pass1", ""),
                    row.get("livecodebench_pass1", ""),
                    row.get("num_generated", ""),
                    row.get("num_after_filter", ""),
                    row.get("timestamp", ""),
                ]
            print(",".join(values))

    print("")
    print("SLURM queue:")
    for line in queue_lines():
        print(line)


if __name__ == "__main__":
    main()
