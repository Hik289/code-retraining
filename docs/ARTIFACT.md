# Artifact Guide

This guide maps the public `code-retraining` repository to a reviewer-friendly artifact workflow for `When AI Reviews Its Own Code: Recursive Self-Training Collapse in Code LLMs`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `configs/`: Configuration files for model, benchmark, or experiment settings.
- `assets/`: README and paper-facing visual assets.
- Root-level entry points: `fim.py`, `train.py`.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
bash scripts/smoke_test.sh
bash scripts/smoke_loop.sh
bash scripts/smoke_test_binary_filter.sh
bash scripts/smoke_test_filters.sh
```

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python fim.py`
- `bash scripts/debug_completion.sh`
- `bash scripts/debug_completion2.sh`
- `bash scripts/eval_evalplus.sh`
- `bash scripts/eval_livecodebench.sh`
- `python scripts/generate_data.py`
- `bash scripts/generate_data.sh`
- `python scripts/generate_data_filtered.py`
- `bash scripts/run_exec_passrate_analysis.sh`
- `bash scripts/run_selfplay_binary_filter.sh`
- `bash scripts/run_selfplay_compile_filter.sh`
- `bash scripts/run_selfplay_loop.sh`
- `bash scripts/run_selfplay_ppl_filter.sh`
- `bash scripts/run_selfplay_quality_filter.sh`

## Figure Assets

- `assets/gated_retraining_pipeline.jpg`
- `assets/ungated_self_training_loop.jpg`

## Data, Credentials, And Generated Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
