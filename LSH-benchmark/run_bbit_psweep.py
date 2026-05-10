from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
OUTPUT_SUMMARY = BASE_DIR / "bbit_psweep_summary.csv"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def run_tag(threshold: float, num_perm: int, b_bits: int) -> str:
    return f"minhash_bbit_lsh_{threshold:.1f}_{num_perm}_b{b_bits}"


def result_paths(dataset: str, threshold: float, num_perm: int, b_bits: int) -> tuple[Path, Path, Path]:
    result_dir = BASE_DIR / dataset / "minhash_bbit_lsh_results"
    tag = run_tag(threshold, num_perm, b_bits)
    return (
        result_dir / f"{tag}_score.csv",
        result_dir / f"{tag}_stats.json",
        result_dir / f"{tag}_preds.csv",
    )


def read_score(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_stats(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_command(args: argparse.Namespace, dataset: str, threshold: float, b_bits: int) -> list[str]:
    cmd = [
        PYTHON,
        "dedup/minhash_bbit_lsh/bbit_lsh.py",
        "--input",
        dataset,
        "--threshold",
        f"{threshold:.1f}",
        "--num-perm",
        str(args.num_perm),
        "--b-bits",
        str(b_bits),
        "--shingle-size",
        str(args.shingle_size),
        "--max-bucket-size",
        str(args.max_bucket_size),
    ]
    if args.num_bands is not None:
        cmd.extend(["--num-bands", str(args.num_bands)])
    if args.rows_per_band is not None:
        cmd.extend(["--rows-per-band", str(args.rows_per_band)])
    if args.force_compute_minhash:
        cmd.append("--force-compute-minhash")
    return cmd


def run_case(args: argparse.Namespace, env: dict[str, str], dataset: str, threshold: float, b_bits: int) -> tuple[str, float, int, int]:
    cmd = build_command(args, dataset, threshold, b_bits)
    print("\n==>", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    return dataset, threshold, b_bits, completed.returncode


def collect_row(args: argparse.Namespace, p: float, dataset: str, threshold: float, b_bits: int) -> dict[str, Any] | None:
    score_path, stats_path, _ = result_paths(dataset, threshold, args.num_perm, b_bits)
    if not score_path.exists() or not stats_path.exists():
        print(f"[warn] missing output for {dataset} threshold={threshold:.1f} b_bits={b_bits}")
        return None

    score = read_score(score_path)
    stats = read_stats(stats_path)
    return {
        "dataset": dataset,
        "duplicate_prevalence": p,
        "threshold": threshold,
        "b_bits": b_bits,
        "num_perm": args.num_perm,
        "shingle_size": args.shingle_size,
        "precision": score.get("precision", ""),
        "recall": score.get("recall", ""),
        "f1": score.get("f1", ""),
        "auc_roc": score.get("auc_roc", ""),
        "acc": score.get("acc", ""),
        "bal_acc": score.get("bal_acc", ""),
        "tp": score.get("tp", ""),
        "fp": score.get("fp", ""),
        "fn": score.get("fn", ""),
        "candidate_pairs": stats.get("candidate_pairs", ""),
        "inserted_docs": stats.get("inserted_docs", ""),
        "predicted_duplicates": stats.get("predicted_duplicates", ""),
        "index_bucket_count": stats.get("index_bucket_count", ""),
        "index_entry_count": stats.get("index_entry_count", ""),
        "skipped_large_buckets": stats.get("skipped_large_buckets", ""),
        "full_signature_bytes": stats.get("full_signature_bytes", ""),
        "bbit_signature_bytes": stats.get("bbit_signature_bytes", ""),
        "estimated_index_bytes": stats.get("estimated_index_bytes", ""),
        "peak_memory_mb": stats.get("peak_memory_mb", ""),
        "runtime_sec": stats.get("runtime_sec", ""),
        "signature_sec": stats.get("signature_sec", ""),
        "query_sec": stats.get("query_sec", ""),
        "insert_sec": stats.get("insert_sec", ""),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run b-bit MinHashLSH sweep on benchmark_dfs.")
    parser.add_argument("--datasets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--b-bits", default="8", help="Comma-separated b-bit values, e.g. 4,8,16.")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--shingle-size", type=int, default=1)
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--rows-per-band", type=int, default=None)
    parser.add_argument("--max-bucket-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers. Runs at most one threshold per dataset at a time.")
    parser.add_argument("--force", action="store_true", help="Re-run even if score and stats already exist.")
    parser.add_argument("--force-compute-minhash", action="store_true")
    parser.add_argument("--summary-out", default=str(OUTPUT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = get_args()
    dataset_ps = parse_float_list(args.datasets)
    thresholds = parse_float_list(args.thresholds)
    b_bits_values = parse_int_list(args.b_bits)
    rows: list[dict[str, object]] = []
    missing_cases: list[tuple[float, str, float, int]] = []
    completed_cases: list[tuple[float, str, float, int]] = []

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    for p in dataset_ps:
        dataset = dataset_tag(p)
        jsonl_path = BASE_DIR / "benchmark_dfs" / f"{dataset}.jsonl"
        csv_path = BASE_DIR / "benchmark_dfs" / f"{dataset}.csv"
        if not jsonl_path.exists() or not csv_path.exists():
            print(f"[skip] missing benchmark files for {dataset}")
            continue

        for b_bits in b_bits_values:
            for threshold in thresholds:
                score_path, stats_path, _ = result_paths(dataset, threshold, args.num_perm, b_bits)
                if not args.force and score_path.exists() and stats_path.exists():
                    print(f"[reuse] {dataset} threshold={threshold:.1f} b_bits={b_bits}")
                    completed_cases.append((p, dataset, threshold, b_bits))
                else:
                    missing_cases.append((p, dataset, threshold, b_bits))

    if missing_cases:
        workers = max(1, int(args.num_workers))
        if workers == 1:
            for p, dataset, threshold, b_bits in missing_cases:
                _, _, _, returncode = run_case(args, env, dataset, threshold, b_bits)
                if returncode != 0:
                    print(f"[error] failed: {dataset} threshold={threshold:.1f} b_bits={b_bits}")
                    return returncode
                completed_cases.append((p, dataset, threshold, b_bits))
        else:
            print(f"[parallel] num_workers={workers}; at most one active case per dataset/b_bits cache")
            pending = list(missing_cases)
            active_keys: set[tuple[str, int]] = set()
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while pending or futures:
                    while pending and len(futures) < workers:
                        selected_index = None
                        for i, (_, dataset, _, b_bits) in enumerate(pending):
                            key = (dataset, b_bits)
                            if key not in active_keys:
                                selected_index = i
                                break
                        if selected_index is None:
                            break
                        p, dataset, threshold, b_bits = pending.pop(selected_index)
                        active_keys.add((dataset, b_bits))
                        future = executor.submit(run_case, args, env, dataset, threshold, b_bits)
                        futures[future] = (p, dataset, threshold, b_bits)

                    if not futures:
                        continue

                    for future in as_completed(list(futures.keys()), timeout=None):
                        p, dataset, threshold, b_bits = futures.pop(future)
                        active_keys.discard((dataset, b_bits))
                        _, _, _, returncode = future.result()
                        if returncode != 0:
                            print(f"[error] failed: {dataset} threshold={threshold:.1f} b_bits={b_bits}")
                            return returncode
                        completed_cases.append((p, dataset, threshold, b_bits))
                        break

    for p in dataset_ps:
        dataset = dataset_tag(p)
        for b_bits in b_bits_values:
            for threshold in thresholds:
                if (p, dataset, threshold, b_bits) not in completed_cases:
                    continue
                row = collect_row(args, p, dataset, threshold, b_bits)
                if row is not None:
                    rows.append(row)

    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = BASE_DIR / summary_path
    if rows:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[done] wrote summary: {summary_path}")
    else:
        print("[done] no rows collected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
