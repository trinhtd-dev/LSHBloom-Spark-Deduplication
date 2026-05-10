from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
OUTPUT_SUMMARY = BASE_DIR / "lsh_forest_psweep_summary.csv"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def run_tag(threshold: float, num_perm: int, num_trees: int, top_k: int) -> str:
    return f"lsh_forest_{threshold:.1f}_{num_perm}_l{num_trees}_top{top_k}"


def result_paths(dataset: str, threshold: float, num_perm: int, num_trees: int, top_k: int) -> tuple[Path, Path]:
    result_dir = BASE_DIR / dataset / "lsh_forest_results"
    tag = run_tag(threshold, num_perm, num_trees, top_k)
    return (
        result_dir / f"{tag}_score.csv",
        result_dir / f"{tag}_stats.json",
    )


def read_score(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_stats(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash LSH Forest sweep on benchmark_dfs.")
    parser.add_argument("--datasets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-trees", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--shingle-size", type=int, default=1)
    parser.add_argument("--index-batch-size", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Re-run even if score and stats already exist.")
    parser.add_argument("--force-compute-minhash", action="store_true")
    parser.add_argument("--summary-out", default=str(OUTPUT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = get_args()
    dataset_ps = parse_float_list(args.datasets)
    thresholds = parse_float_list(args.thresholds)
    rows: list[dict[str, object]] = []

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    for p in dataset_ps:
        dataset = dataset_tag(p)
        jsonl_path = BASE_DIR / "benchmark_dfs" / f"{dataset}.jsonl"
        csv_path = BASE_DIR / "benchmark_dfs" / f"{dataset}.csv"
        if not jsonl_path.exists() or not csv_path.exists():
            print(f"[skip] missing benchmark files for {dataset}")
            continue

        for threshold in thresholds:
            score_path, stats_path = result_paths(dataset, threshold, args.num_perm, args.num_trees, args.top_k)
            if not args.force and score_path.exists() and stats_path.exists():
                print(f"[reuse] {dataset} threshold={threshold:.1f}")
            else:
                cmd = [
                    PYTHON,
                    "dedup/lsh_forest/lsh_forest.py",
                    "--input",
                    dataset,
                    "--threshold",
                    f"{threshold:.1f}",
                    "--num-perm",
                    str(args.num_perm),
                    "--num-trees",
                    str(args.num_trees),
                    "--top-k",
                    str(args.top_k),
                    "--shingle-size",
                    str(args.shingle_size),
                    "--index-batch-size",
                    str(args.index_batch_size),
                ]
                if args.force_compute_minhash:
                    cmd.append("--force-compute-minhash")

                print("\n==>", " ".join(cmd))
                completed = subprocess.run(cmd, cwd=BASE_DIR, env=env)
                if completed.returncode != 0:
                    print(f"[error] failed: {dataset} threshold={threshold:.1f}")
                    return completed.returncode

            if not score_path.exists() or not stats_path.exists():
                print(f"[warn] missing output for {dataset} threshold={threshold:.1f}")
                continue

            score = read_score(score_path)
            stats = read_stats(stats_path)
            rows.append(
                {
                    "dataset": dataset,
                    "duplicate_prevalence": p,
                    "threshold": threshold,
                    "num_perm": args.num_perm,
                    "num_trees": args.num_trees,
                    "top_k": args.top_k,
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
                    "index_rebuilds": stats.get("index_rebuilds", ""),
                    "index_bucket_count": stats.get("index_bucket_count", ""),
                    "index_entry_count": stats.get("index_entry_count", ""),
                    "signature_bytes": stats.get("signature_bytes", ""),
                    "estimated_index_bytes": stats.get("estimated_index_bytes", ""),
                    "peak_memory_mb": stats.get("peak_memory_mb", ""),
                    "runtime_sec": stats.get("runtime_sec", ""),
                    "signature_sec": stats.get("signature_sec", ""),
                    "query_sec": stats.get("query_sec", ""),
                    "insert_sec": stats.get("insert_sec", ""),
                    "index_sec": stats.get("index_sec", ""),
                }
            )

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
