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
OUTPUT_SUMMARY = BENCHMARK_ROOT / "lsh_vs_fast_minhash_lsh_compare.csv"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def pure_lsh_score_path(dataset: str, threshold: float, num_perm: int) -> Path:
    return BENCHMARK_ROOT / dataset / "lsh_results" / f"lsh_{threshold:.1f}_{num_perm}_score.csv"


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


def run_pure_lsh(args: argparse.Namespace, env: dict[str, str], dataset: str, threshold: float) -> int:
    cmd = [
        PYTHON,
        "dedup/lsh/lsh.py",
        "--input",
        dataset,
        "--sim-threshold",
        f"{threshold:.1f}",
        "--num-perm",
        str(args.num_perm),
        "--ngram",
        str(args.ngram),
    ]
    if args.force_compute_minhash:
        cmd.append("--force-compute-minhash")
    print("\n==>", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=BENCHMARK_ROOT, env=env)
    return completed.returncode


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pure MinHashLSH with fast rolling hash LSH.")
    parser.add_argument("--datasets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-run pure LSH even if score exists.")
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
        jsonl_path = BENCHMARK_ROOT / "benchmark_dfs" / f"{dataset}.jsonl"
        csv_path = BENCHMARK_ROOT / "benchmark_dfs" / f"{dataset}.csv"
        if not jsonl_path.exists() or not csv_path.exists():
            print(f"[skip] missing benchmark files for {dataset}")
            continue
        for threshold in thresholds:
            if pure_lsh_score_path(dataset, threshold, args.num_perm).exists() and not args.force:
                print(f"[reuse] pure lsh {dataset} threshold={threshold:.1f}")
                completed.add((p, dataset, threshold))
            elif not args.collect_only:
                pending.append((p, dataset, threshold))

    if pending and not args.collect_only:
        workers = max(1, int(args.num_workers))
        if workers == 1:
            for p, dataset, threshold in pending:
                rc = run_pure_lsh(args, env, dataset, threshold)
                if rc != 0:
                    print(f"[error] pure lsh failed: {dataset} threshold={threshold:.1f}")
                    return rc
                completed.add((p, dataset, threshold))
        else:
            print(f"[parallel] num_workers={workers}; at most one active pure-lsh case per dataset")
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
                        future = executor.submit(run_pure_lsh, args, env, dataset, threshold)
                        futures[future] = (p, dataset, threshold)

                    if not futures:
                        continue

                    for future in as_completed(list(futures.keys()), timeout=None):
                        p, dataset, threshold = futures.pop(future)
                        active_datasets.discard(dataset)
                        rc = future.result()
                        if rc != 0:
                            print(f"[error] pure lsh failed: {dataset} threshold={threshold:.1f}")
                            return rc
                        completed.add((p, dataset, threshold))
                        break

    rows: list[dict[str, object]] = []
    for p in dataset_ps:
        dataset = dataset_tag(p)
        for threshold in thresholds:
            pure_path = pure_lsh_score_path(dataset, threshold, args.num_perm)
            fast_path = fast_score_path(dataset, threshold, args.num_perm)
            if not pure_path.exists() or not fast_path.exists():
                print(f"[warn] missing pair: {dataset} threshold={threshold:.1f}")
                continue
            pure = read_score(pure_path)
            fast = read_score(fast_path)
            row = {
                "dataset": dataset,
                "duplicate_prevalence": f"{p:.1f}",
                "threshold": f"{threshold:.1f}",
                "num_perm": args.num_perm,
                "lsh_precision": pure.get("precision", ""),
                "fast_precision": fast.get("precision", ""),
                "delta_precision": float(fast["precision"]) - float(pure["precision"]),
                "lsh_recall": pure.get("recall", ""),
                "fast_recall": fast.get("recall", ""),
                "delta_recall": float(fast["recall"]) - float(pure["recall"]),
                "lsh_f1": pure.get("f1", ""),
                "fast_f1": fast.get("f1", ""),
                "delta_f1": float(fast["f1"]) - float(pure["f1"]),
                "lsh_tp": pure.get("tp", ""),
                "fast_tp": fast.get("tp", ""),
                "lsh_fp": pure.get("fp", ""),
                "fast_fp": fast.get("fp", ""),
                "lsh_fn": pure.get("fn", ""),
                "fast_fn": fast.get("fn", ""),
            }
            rows.append(row)

    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = BENCHMARK_ROOT / summary_path
    if rows:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[done] wrote comparison: {summary_path}")
    else:
        print("[done] no rows collected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
