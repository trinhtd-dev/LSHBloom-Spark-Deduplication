from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = REPO_ROOT / "LSH-benchmark"
PYTHON = sys.executable
OUTPUT_SUMMARY = BENCHMARK_ROOT / "char_minhash_lsh_vs_fast_compare.csv"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def char_score_path(dataset: str, threshold: float, num_perm: int) -> Path:
    return (
        BENCHMARK_ROOT
        / dataset
        / "char_minhash_lsh_results"
        / f"char_minhash_lsh_{threshold:.1f}_{num_perm}_score.csv"
    )


def fast_score_path(dataset: str, threshold: float, num_perm: int) -> Path:
    return (
        BENCHMARK_ROOT
        / dataset
        / "fast_minhash_lsh_results"
        / f"fast_minhash_lsh_{threshold:.1f}_{num_perm}_score.csv"
    )


def read_score(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def run_char_lsh(args: argparse.Namespace, env: dict[str, str], dataset: str, threshold: float) -> int:
    cmd = [
        PYTHON,
        "hashing/char_minhash_lsh_benchmark.py",
        "--input",
        dataset,
        "--sim-threshold",
        f"{threshold:.1f}",
        "--num-perm",
        str(args.num_perm),
        "--shingle-size",
        str(args.shingle_size),
        "--max-shingles",
        str(args.max_shingles),
    ]
    if args.force_compute_minhash:
        cmd.append("--force-compute-minhash")
    print("\n==>", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standard char MinHashLSH with fast rolling hash LSH.")
    parser.add_argument("--datasets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-compute-minhash", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--summary-out", default=str(OUTPUT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = get_args()
    dataset_ps = parse_float_list(args.datasets)
    thresholds = parse_float_list(args.thresholds)
    pending: list[tuple[float, str, float]] = []
    completed: set[tuple[float, str, float]] = set()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    for p in dataset_ps:
        dataset = dataset_tag(p)
        if not (BENCHMARK_ROOT / "benchmark_dfs" / f"{dataset}.jsonl").exists():
            print(f"[skip] missing benchmark files for {dataset}")
            continue
        for threshold in thresholds:
            if char_score_path(dataset, threshold, args.num_perm).exists() and not args.force:
                print(f"[reuse] char lsh {dataset} threshold={threshold:.1f}")
                completed.add((p, dataset, threshold))
            elif not args.collect_only:
                pending.append((p, dataset, threshold))

    workers = max(1, int(args.num_workers))
    if pending and not args.collect_only:
        if workers == 1:
            for p, dataset, threshold in pending:
                rc = run_char_lsh(args, env, dataset, threshold)
                if rc != 0:
                    return rc
                completed.add((p, dataset, threshold))
        else:
            active_datasets: set[str] = set()
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while pending or futures:
                    while pending and len(futures) < workers:
                        selected_index = None
                        for index, (_, dataset, _) in enumerate(pending):
                            if dataset not in active_datasets:
                                selected_index = index
                                break
                        if selected_index is None:
                            break
                        p, dataset, threshold = pending.pop(selected_index)
                        active_datasets.add(dataset)
                        future = executor.submit(run_char_lsh, args, env, dataset, threshold)
                        futures[future] = (p, dataset, threshold)
                    if not futures:
                        continue
                    for future in as_completed(list(futures.keys())):
                        p, dataset, threshold = futures.pop(future)
                        active_datasets.discard(dataset)
                        rc = future.result()
                        if rc != 0:
                            return rc
                        completed.add((p, dataset, threshold))
                        break

    rows: list[dict[str, object]] = []
    for p in dataset_ps:
        dataset = dataset_tag(p)
        for threshold in thresholds:
            cpath = char_score_path(dataset, threshold, args.num_perm)
            fpath = fast_score_path(dataset, threshold, args.num_perm)
            if not cpath.exists() or not fpath.exists():
                print(f"[warn] missing pair: {dataset} threshold={threshold:.1f}")
                continue
            char = read_score(cpath)
            fast = read_score(fpath)
            rows.append(
                {
                    "dataset": dataset,
                    "duplicate_prevalence": f"{p:.1f}",
                    "threshold": f"{threshold:.1f}",
                    "num_perm": args.num_perm,
                    "char_precision": char.get("precision", ""),
                    "fast_precision": fast.get("precision", ""),
                    "delta_precision": float(fast["precision"]) - float(char["precision"]),
                    "char_recall": char.get("recall", ""),
                    "fast_recall": fast.get("recall", ""),
                    "delta_recall": float(fast["recall"]) - float(char["recall"]),
                    "char_f1": char.get("f1", ""),
                    "fast_f1": fast.get("f1", ""),
                    "delta_f1": float(fast["f1"]) - float(char["f1"]),
                }
            )

    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = BENCHMARK_ROOT / summary_path
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[done] wrote comparison: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
