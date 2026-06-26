#!/usr/bin/env python3
"""Aggregate formal self-play results and render dataset-specific SVG plots."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


MODELS = ["santacoder", "starcoder2", "qwen25", "codellama"]
FILTERS = ["none", "compile", "quality", "ppl", "binary"]
CSV_COLUMNS = [
    "model",
    "filter",
    "round",
    "steps_total",
    "humaneval_pass1",
    "humaneval_plus_pass1",
    "mbpp_pass1",
    "mbpp_plus_pass1",
    "livecodebench_pass1",
    "train_loss",
    "num_generated",
    "num_after_filter",
    "filter_pass_rate",
    "generation_time_sec",
    "training_time_sec",
    "eval_time_sec",
    "timestamp",
]
PLOT_METRICS = [
    ("humaneval_plus_pass1", "HumanEval+ pass@1", "01_humaneval_plus_local_scale.svg"),
    ("mbpp_plus_pass1", "MBPP+ pass@1", "02_mbpp_plus_local_scale.svg"),
    ("livecodebench_pass1", "LiveCodeBench pass@1", "03_livecodebench_local_scale.svg"),
]
BASELINES = {
    "santacoder": {
        "humaneval_plus_pass1": 0.1707,
        "mbpp_plus_pass1": 0.2937,
        "livecodebench_pass1": 0.0175,
    },
    "starcoder2": {
        "humaneval_plus_pass1": 0.2744,
        "mbpp_plus_pass1": 0.4921,
        "livecodebench_pass1": 0.0850,
    },
    "qwen25": {
        "humaneval_plus_pass1": 0.3720,
        "mbpp_plus_pass1": 0.5820,
        "livecodebench_pass1": 0.2375,
    },
    "codellama": {
        "humaneval_plus_pass1": 0.2500,
        "mbpp_plus_pass1": 0.4206,
        "livecodebench_pass1": 0.0700,
    },
}
COLORS = {
    "none": "#111827",
    "compile": "#2563eb",
    "quality": "#059669",
    "ppl": "#d97706",
    "binary": "#dc2626",
}
MODEL_LABELS = {
    "santacoder": "SantaCoder",
    "starcoder2": "StarCoder2",
    "qwen25": "Qwen2.5-Coder",
    "codellama": "Code Llama",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--max-round", type=int, default=5)
    return parser.parse_args()


def sort_key(key: tuple[str, str, int]) -> tuple[int, int, int]:
    model, filter_name, round_number = key
    return MODELS.index(model), FILTERS.index(filter_name), round_number


def load_rows(results_dir: Path, max_round: int) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for path in sorted(results_dir.glob("*/*/results.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError(f"Unexpected CSV schema in {path}")
            for row in reader:
                round_number = int(row["round"])
                if round_number > max_round:
                    continue
                key = (row["model"], row["filter"], round_number)
                if key in by_key:
                    if by_key[key] != row:
                        raise ValueError(f"Conflicting duplicate row for {key} in {path}")
                    print(f"Warning: ignored identical duplicate row for {key}")
                    continue
                by_key[key] = row

    expected = {
        (model, filter_name, round_number)
        for model in MODELS
        for filter_name in FILTERS
        for round_number in range(1, max_round + 1)
    }
    missing = sorted(expected - set(by_key))
    unexpected = sorted(set(by_key) - expected)
    if missing:
        raise ValueError(f"Missing formal result rows: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected formal result rows: {unexpected}")
    return [by_key[key] for key in sorted(by_key, key=sort_key)]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {(row["model"], row["filter"], int(row["round"])): row for row in rows}


def write_round_summary(
    path: Path, indexed: dict[tuple[str, str, int], dict[str, str]], max_round: int
) -> None:
    columns = [
        "model",
        "filter",
        "round",
        "humaneval_plus_pass1",
        "mbpp_plus_pass1",
        "livecodebench_pass1",
        "humaneval_plus_retention",
        "mbpp_plus_retention",
        "livecodebench_retention",
    ]
    output = []
    for model in MODELS:
        for filter_name in FILTERS:
            row = indexed[(model, filter_name, max_round)]
            he_plus = float(row["humaneval_plus_pass1"])
            mbpp_plus = float(row["mbpp_plus_pass1"])
            lcb = float(row["livecodebench_pass1"])
            baseline = BASELINES[model]
            output.append(
                {
                    "model": model,
                    "filter": filter_name,
                    "round": max_round,
                    "humaneval_plus_pass1": f"{he_plus:.4f}",
                    "mbpp_plus_pass1": f"{mbpp_plus:.4f}",
                    "livecodebench_pass1": f"{lcb:.4f}",
                    "humaneval_plus_retention": f"{he_plus / baseline['humaneval_plus_pass1']:.4f}",
                    "mbpp_plus_retention": f"{mbpp_plus / baseline['mbpp_plus_pass1']:.4f}",
                    "livecodebench_retention": f"{lcb / baseline['livecodebench_pass1']:.4f}",
                }
            )
    write_csv(path, columns, output)


def first_below_half(
    indexed: dict[tuple[str, str, int], dict[str, str]],
    model: str,
    filter_name: str,
    metric: str,
    max_round: int,
) -> str:
    threshold = BASELINES[model][metric] * 0.5
    for round_number in range(1, max_round + 1):
        if float(indexed[(model, filter_name, round_number)][metric]) < threshold:
            return str(round_number)
    return ""


def write_collapse_speed(
    path: Path, indexed: dict[tuple[str, str, int], dict[str, str]], max_round: int
) -> None:
    tracked = [metric for metric, _, _ in PLOT_METRICS]
    columns = ["model", "filter"] + [f"{metric}_below_50pct_round" for metric in tracked]
    output = []
    for model in MODELS:
        for filter_name in FILTERS:
            row: dict[str, object] = {"model": model, "filter": filter_name}
            for metric in tracked:
                row[f"{metric}_below_50pct_round"] = first_below_half(
                    indexed, model, filter_name, metric, max_round
                )
            output.append(row)
    write_csv(path, columns, output)


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attributes = " ".join(f'{name.replace("_", "-")}="{value}"' for name, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(text)}</text>'


def chart_x(x0: float, width: float, index: int, count: int) -> float:
    return x0 + width * index / max(1, count - 1)


def chart_y(y0: float, height: float, value: float, y_min: float, y_max: float) -> float:
    return y0 + height * (y_max - value) / (y_max - y_min)


def decimals_for_axis(y_max: float) -> int:
    if y_max <= 0.01:
        return 4
    if y_max <= 0.1:
        return 3
    return 2


def local_bounds(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = max(high - low, 0.01)
    return max(0.0, low - span * 0.18), high + span * 0.18


def add_axes(
    svg: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_labels: list[str],
    y_min: float,
    y_max: float,
) -> None:
    decimals = decimals_for_axis(y_max)
    svg.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#f9fafb" stroke="#d1d5db"/>')
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = chart_y(y0, height, value, y_min, y_max)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(svg_text(x0 - 10, y + 5, f"{value:.{decimals}f}", text_anchor="end", font_size=12, fill="#6b7280"))
    for index, label in enumerate(x_labels):
        x = chart_x(x0, width, index, len(x_labels))
        svg.append(svg_text(x, y0 + height + 22, label, text_anchor="middle", font_size=12, fill="#6b7280"))


def add_line(
    svg: list[str],
    values: list[float],
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> None:
    points = []
    for index, value in enumerate(values):
        x = chart_x(x0, width, index, len(values))
        y = chart_y(y0, height, value, y_min, y_max)
        points.append(f"{x:.1f},{y:.1f}")
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for point in points:
        x, y = point.split(",")
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')


def add_legend(svg: list[str], y: float) -> None:
    for index, filter_name in enumerate(FILTERS):
        x = 185 + index * 175
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{COLORS[filter_name]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{x + 14}" cy="{y}" r="4" fill="{COLORS[filter_name]}"/>')
        svg.append(svg_text(x + 38, y + 5, filter_name, font_size=14, fill="#374151"))


def render_metric_local_scale(
    path: Path,
    indexed: dict[tuple[str, str, int], dict[str, str]],
    metric: str,
    title: str,
    max_round: int,
) -> None:
    width, height = 1200, 870
    panel_w, panel_h = 500, 270
    origin_x, origin_y = 100, 145
    gap_x, gap_y = 80, 105
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(50, 45, f"{title} trajectories by filter", font_size=26, font_weight="bold", fill="#111827"),
        svg_text(50, 75, "Each model uses its own y-axis; base= shows performance before self-play", font_size=16, fill="#4b5563"),
    ]
    for model_index, model in enumerate(MODELS):
        col, row_index = model_index % 2, model_index // 2
        x0 = origin_x + col * (panel_w + gap_x)
        y0 = origin_y + row_index * (panel_h + gap_y)
        all_values = [
            float(indexed[(model, filter_name, round_number)][metric])
            for filter_name in FILTERS
            for round_number in range(1, max_round + 1)
        ]
        y_min, y_max = local_bounds(all_values)
        base = BASELINES[model][metric]
        base_decimals = 4 if base < 0.1 else 3
        svg.append(svg_text(x0, y0 - 24, MODEL_LABELS[model], font_size=18, font_weight="bold", fill="#111827"))
        svg.append(svg_text(x0 + panel_w, y0 - 24, f"base={base:.{base_decimals}f}", text_anchor="end", font_size=13, fill="#6b7280"))
        add_axes(svg, x0, y0, panel_w, panel_h, [str(round_number) for round_number in range(1, max_round + 1)], y_min, y_max)
        for filter_name in FILTERS:
            values = [
                float(indexed[(model, filter_name, round_number)][metric])
                for round_number in range(1, max_round + 1)
            ]
            add_line(svg, values, COLORS[filter_name], x0, y0, panel_w, panel_h, y_min, y_max)
    svg.append(svg_text(width / 2, height - 67, "self-play round", text_anchor="middle", font_size=14, fill="#4b5563"))
    add_legend(svg, height - 35)
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def clean_old_plots(output_dir: Path) -> None:
    for pattern in ("*.svg", "*.png"):
        for path in output_dir.glob(pattern):
            path.unlink()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir, args.max_round)
    indexed = index_rows(rows)
    write_csv(args.output_dir / "all_results.csv", CSV_COLUMNS, rows)
    write_round_summary(args.output_dir / "round5_summary.csv", indexed, args.max_round)
    write_collapse_speed(args.output_dir / "collapse_speed.csv", indexed, args.max_round)
    clean_old_plots(args.output_dir)
    for metric, title, filename in PLOT_METRICS:
        render_metric_local_scale(args.output_dir / filename, indexed, metric, title, args.max_round)
    print(f"Wrote {len(rows)} formal rows and {len(PLOT_METRICS)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
