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
OUTPUT_SUMMARY = BENCHMARK_ROOT / "fast_minhash_lsh_psweep_summary.csv"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def result_paths(dataset: str, threshold: float, num_perm: int) -> tuple[Path, Path, Path]:
    result_dir = BENCHMARK_ROOT / dataset / "fast_minhash_lsh_results"
    tag = f"fast_minhash_lsh_{threshold:.1f}_{num_perm}"
    return (
        result_dir / f"{tag}_score.csv",
        result_dir / f"{tag}_preds.csv",
        result_dir / f"{tag}_meta.csv",
    )


def read_score(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["key"]: row["value"] for row in csv.DictReader(f)}


def build_command(args: argparse.Namespace, dataset: str, threshold: float) -> list[str]:
    cmd = [
        PYTHON,
        "hashing/fast_minhash_lsh_benchmark.py",
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
    if args.bands is not None or args.rows is not None:
        if args.bands is None or args.rows is None:
            raise ValueError("--bands and --rows must be set together")
        cmd.extend(["--bands", str(args.bands), "--rows", str(args.rows)])
    if args.verify_signature_threshold:
        cmd.append("--verify-signature-threshold")
    if args.force_compute_minhash:
        cmd.append("--force-compute-minhash")
    return cmd


def run_case(args: argparse.Namespace, env: dict[str, str], dataset: str, threshold: float) -> int:
    cmd = build_command(args, dataset, threshold)
    print("\n==>", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


def collect_row(args: argparse.Namespace, p: float, dataset: str, threshold: float) -> dict[str, object] | None:
    score_path, preds_path, meta_path = result_paths(dataset, threshold, args.num_perm)
    if not score_path.exists():
        print(f"[warn] missing score for {dataset} threshold={threshold:.1f}: {score_path}")
        return None

    score = read_score(score_path)
    meta = read_meta(meta_path)
    return {
        "dataset": dataset,
        "duplicate_prevalence": f"{p:.1f}",
        "threshold": f"{threshold:.1f}",
        "algorithm": "fast_minhash_lsh",
        "num_perm": args.num_perm,
        "bands": meta.get("bands", ""),
        "rows": meta.get("rows", ""),
        "used_perm": meta.get("used_perm", ""),
        "shingle_size": args.shingle_size,
        "max_shingles": args.max_shingles,
        "signature_input_policy": meta.get("signature_input_policy", ""),
        "verify_signature_threshold": args.verify_signature_threshold,
        "precision": score.get("precision", ""),
        "recall": score.get("recall", ""),
        "f1": score.get("f1", ""),
        "auc_roc": score.get("auc_roc", ""),
        "acc": score.get("acc", ""),
        "bal_acc": score.get("bal_acc", ""),
        "tp": score.get("tp", ""),
        "fp": score.get("fp", ""),
        "fn": score.get("fn", ""),
        "preds_file": str(preds_path),
        "score_file": str(score_path),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full p/threshold sweep for hashing/fast_minhash_lsh_benchmark.py."
    )
    parser.add_argument("--datasets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=150)
    parser.add_argument("--bands", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--verify-signature-threshold", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even when score files already exist.")
    parser.add_argument("--force-compute-minhash", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing score files.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers. At most one threshold per dataset runs at a time.",
    )
    parser.add_argument("--summary-out", default=str(OUTPUT_SUMMARY))
    return parser.parse_args()


def main() -> int:
    args = get_args()
    dataset_ps = parse_float_list(args.datasets)
    thresholds = parse_float_list(args.thresholds)
    completed_cases: set[tuple[float, str, float]] = set()
    pending_cases: list[tuple[float, str, float]] = []

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
            score_path, _, _ = result_paths(dataset, threshold, args.num_perm)
            if score_path.exists() and not args.force:
                print(f"[reuse] {dataset} threshold={threshold:.1f}")
                completed_cases.add((p, dataset, threshold))
                continue
            if args.collect_only:
                continue

            pending_cases.append((p, dataset, threshold))

    if pending_cases and not args.collect_only:
        workers = max(1, int(args.num_workers))
        if workers == 1:
            for p, dataset, threshold in pending_cases:
                returncode = run_case(args, env, dataset, threshold)
                if returncode != 0:
                    print(f"[error] failed: {dataset} threshold={threshold:.1f}")
                    return returncode
                completed_cases.add((p, dataset, threshold))
        else:
            print(f"[parallel] num_workers={workers}; at most one active case per dataset")
            pending = list(pending_cases)
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
                        future = executor.submit(run_case, args, env, dataset, threshold)
                        futures[future] = (p, dataset, threshold)

                    if not futures:
                        continue

                    for future in as_completed(list(futures.keys()), timeout=None):
                        p, dataset, threshold = futures.pop(future)
                        active_datasets.discard(dataset)
                        returncode = future.result()
                        if returncode != 0:
                            print(f"[error] failed: {dataset} threshold={threshold:.1f}")
                            return returncode
                        completed_cases.add((p, dataset, threshold))
                        break

    rows: list[dict[str, object]] = []
    for p in dataset_ps:
        dataset = dataset_tag(p)
        for threshold in thresholds:
            if (p, dataset, threshold) not in completed_cases and not args.collect_only:
                continue
            row = collect_row(args, p, dataset, threshold)
            if row is not None:
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
        print(f"\n[done] wrote summary: {summary_path}")
    else:
        print("[done] no rows collected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
