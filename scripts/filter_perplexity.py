import argparse
import json
import math

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_percent", type=float, default=25,
                        help="保留 PPL 最低的前 N%%")
    args = parser.parse_args()

    samples = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    valid = [(i, s) for i, s in enumerate(samples) if math.isfinite(s.get("ppl", float("inf")))]
    invalid_count = len(samples) - len(valid)

    valid.sort(key=lambda x: x[1]["ppl"])

    keep_count = max(1, int(len(valid) * args.top_percent / 100))
    kept = valid[:keep_count]

    with open(args.output_file, "w") as f:
        for _, sample in kept:
            out = {k: v for k, v in sample.items() if k != "ppl"}
            f.write(json.dumps(out) + "\n")

    ppls = np.array([s["ppl"] for _, s in valid])
    kept_ppls = np.array([s["ppl"] for _, s in kept])
    print("\n===== PPL 过滤统计 =====")
    print(f"输入: {len(samples)} (有效: {len(valid)}, inf: {invalid_count})")
    print(f"保留: {keep_count} (top {args.top_percent}%)")
    if len(ppls) > 0:
        print(f"全部 PPL: [{ppls.min():.2f}, {np.median(ppls):.2f}, {ppls.max():.2f}]")
    if len(kept_ppls) > 0:
        print(f"保留 PPL: [{kept_ppls.min():.2f}, {np.median(kept_ppls):.2f}, {kept_ppls.max():.2f}]")
    print(f"写入: {args.output_file}")


if __name__ == "__main__":
    main()
