import argparse
import json
import math

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Keep samples with score > threshold (default: 0.0)")
    args = parser.parse_args()

    samples = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    valid = [(i, s) for i, s in enumerate(samples)
             if math.isfinite(s.get("score", float("nan")))]
    invalid_count = len(samples) - len(valid)

    kept = [(i, s) for i, s in valid if s["score"] > args.threshold]

    with open(args.output_file, "w") as f:
        for _, sample in kept:
            out = {k: v for k, v in sample.items() if k != "score"}
            f.write(json.dumps(out) + "\n")

    all_scores = np.array([s["score"] for _, s in valid]) if valid else np.array([])
    kept_scores = np.array([s["score"] for _, s in kept]) if kept else np.array([])
    pass_rate = len(kept) / len(valid) * 100 if valid else 0.0

    print("\n===== Binary Classifier Filter =====")
    print(f"Input:     {len(samples)} (finite: {len(valid)}, nan/inf: {invalid_count})")
    print(f"Threshold: score > {args.threshold}")
    print(f"Kept:      {len(kept)} ({pass_rate:.1f}% pass rate)")
    if len(all_scores) > 0:
        print(f"All scores:  min={all_scores.min():.4f}  median={np.median(all_scores):.4f}  max={all_scores.max():.4f}")
    if len(kept_scores) > 0:
        print(f"Kept scores: min={kept_scores.min():.4f}  median={np.median(kept_scores):.4f}  max={kept_scores.max():.4f}")
    print(f"Written: {args.output_file}")


if __name__ == "__main__":
    main()
