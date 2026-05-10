import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import DEFAULT_DATA_DIR, ROOT
from doph_lsh import run as run_doph
from minhash_lsh import run as run_minhash
from oph_lsh import run as run_oph


RUNNERS = {
    "minhash": run_minhash,
    "oph": run_oph,
    "doph": run_doph,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MinHashLSH, MinHashLSH+OPH, and MinHashLSH+DOPH benchmarks."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(ROOT / "runs"))
    parser.add_argument("--input", action="append", help="Dataset tag, e.g. test_p_0.1. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Run all test_p_*.jsonl datasets.")
    parser.add_argument("--sketch", choices=["minhash", "oph", "doph", "oph_doph", "all"], default="all")
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_job(job):
    sketch, output_root, data_dir, tag, threshold, num_perm, ngram, limit, force = job
    RUNNERS[sketch](
        output_root=output_root,
        data_dir=data_dir,
        tag=tag,
        threshold=threshold,
        num_perm=num_perm,
        ngram=ngram,
        limit=limit,
        force=force,
    )
    return sketch, tag


def read_single_row(path):
    with open(path, newline="", encoding="utf8") as fin:
        rows = list(csv.DictReader(fin))
    return rows[0] if rows else None


def write_summary(output_root, threshold, num_perm, limit):
    output_root = Path(output_root)
    limit_tag = f"_limit_{limit}" if limit > 0 else ""
    summary_file = output_root / f"summary_{threshold}_{num_perm}{limit_tag}.csv"
    rows = []

    for score_file in output_root.glob(f"test_p_*/*/*_{threshold}_{num_perm}{limit_tag}_score.csv"):
        stats_file = score_file.with_name(score_file.name.replace("_score.csv", "_stats.csv"))
        score_row = read_single_row(score_file)
        stats_row = read_single_row(stats_file) if stats_file.exists() else {}
        if not score_row:
            continue
        merged = {}
        merged.update(stats_row or {})
        merged.update(score_row)
        rows.append(merged)

    if not rows:
        print("[warn] no summary rows found")
        return

    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("sketch", r.get("name", ""))))
    fieldnames = [
        "dataset",
        "sketch",
        "name",
        "threshold",
        "num_perm",
        "ngram",
        "docs",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "acc",
        "bal_acc",
        "tp",
        "fp",
        "fn",
        "predicted_duplicates",
        "avg_empty_bins",
        "docs_with_empty_bins",
        "max_empty_bins",
        "wall_sec",
        "signature_sec",
        "query_sec",
        "insert_sec",
        "peak_rss_gb",
        "signature_cache_gb",
        "index_pickle_gb",
    ]
    with open(summary_file, "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {summary_file}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    if args.all:
        tags = sorted(p.stem for p in data_dir.glob("test_p_*.jsonl"))
    else:
        tags = args.input or ["test_p_0.1"]

    if args.sketch == "all":
        sketches = ["minhash", "oph", "doph"]
    elif args.sketch == "oph_doph":
        sketches = ["oph", "doph"]
    else:
        sketches = [args.sketch]

    jobs = [
        (
            sketch,
            output_root,
            data_dir,
            tag,
            args.sim_threshold,
            args.num_perm,
            args.ngram,
            args.limit,
            args.force,
        )
        for tag in tags
        for sketch in sketches
    ]

    if args.workers <= 1:
        for job in jobs:
            run_job(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                sketch, tag = future.result()
                print(f"[finished] {tag} {sketch}")

    write_summary(output_root, args.sim_threshold, args.num_perm, args.limit)


if __name__ == "__main__":
    main()
