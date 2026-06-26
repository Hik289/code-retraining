"""scripts/analyze_exec_passrate.py — analyze exec() pass rate on generated JSONL data

Runs each sample through exec() in a subprocess with a timeout.
Reports pass/fail breakdown by exception type.

Usage:
    python scripts/analyze_exec_passrate.py \
        --input_file selfplay_results/ppl_filter/generated_data/round1_raw.jsonl \
        --timeout 5 \
        --max_samples 500
"""
import argparse
import json
import multiprocessing
import signal
import sys
from collections import Counter


def exec_worker(code: str, result_queue):
    """Run in a child process: exec code and put result into queue."""
    try:
        exec(compile(code, "<string>", "exec"), {})
        result_queue.put(("ok", None))
    except SyntaxError as e:
        result_queue.put(("SyntaxError", type(e).__name__))
    except ImportError as e:
        result_queue.put(("ImportError", type(e).__name__))
    except ModuleNotFoundError as e:
        result_queue.put(("ModuleNotFoundError", type(e).__name__))
    except NameError as e:
        result_queue.put(("NameError", type(e).__name__))
    except AttributeError as e:
        result_queue.put(("AttributeError", type(e).__name__))
    except TypeError as e:
        result_queue.put(("TypeError", type(e).__name__))
    except ValueError as e:
        result_queue.put(("ValueError", type(e).__name__))
    except Exception as e:
        result_queue.put((type(e).__name__, type(e).__name__))


def run_with_timeout(code: str, timeout: int) -> str:
    """Run exec() in subprocess, return result label."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=exec_worker, args=(code, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout"

    if p.exitcode != 0 and q.empty():
        return f"crash(exit={p.exitcode})"

    if not q.empty():
        label, _ = q.get()
        return label

    return f"crash(exit={p.exitcode})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=5,
                        help="Seconds before killing exec subprocess")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples analyzed")
    args = parser.parse_args()

    samples = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Analyzing {len(samples)} samples from {args.input_file}")
    print(f"Timeout: {args.timeout}s per sample")
    print()

    counts = Counter()
    for i, s in enumerate(samples):
        result = run_with_timeout(s["content"], args.timeout)
        counts[result] += 1
        if (i + 1) % 50 == 0:
            total_so_far = sum(counts.values())
            ok_so_far = counts["ok"]
            print(f"  [{i+1}/{len(samples)}] ok so far: {ok_so_far}/{total_so_far} ({100*ok_so_far/total_so_far:.1f}%)")
            sys.stdout.flush()

    total = sum(counts.values())
    print()
    print("===== exec() Results =====")
    print(f"{'Result':<25} {'Count':>7} {'%':>7}")
    print("-" * 42)
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{label:<25} {cnt:>7} {100*cnt/total:>6.1f}%")
    print("-" * 42)
    print(f"{'TOTAL':<25} {total:>7}")
    print()

    ok = counts["ok"]
    timeout = counts["timeout"]
    import_err = counts["ImportError"] + counts["ModuleNotFoundError"]
    name_err = counts["NameError"]
    other = total - ok - timeout - import_err - name_err

    print("===== Summary =====")
    print(f"Pass (ok):                 {ok:>5} ({100*ok/total:.1f}%)")
    print(f"Timeout (infinite loop?):  {timeout:>5} ({100*timeout/total:.1f}%)")
    print(f"ImportError/ModuleNotFound:{import_err:>5} ({100*import_err/total:.1f}%)")
    print(f"NameError:                 {name_err:>5} ({100*name_err/total:.1f}%)")
    print(f"Other errors:              {other:>5} ({100*other/total:.1f}%)")
    print()
    print("If allowing ImportError/ModuleNotFoundError as 'pass':")
    relaxed = ok + import_err
    print(f"  Relaxed pass rate: {relaxed}/{total} ({100*relaxed/total:.1f}%)")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
