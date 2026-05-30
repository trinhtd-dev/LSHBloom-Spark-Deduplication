from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_pes2o import make_spark, parse_sizes, reset_dir, should_use_pyarrow_io
from benchmark_two_pass_recovery import write_df
from spark_lshbloom import (
    detect_suspect_keys,
    emit_band_rows,
    recover_all_candidate_pairs,
    recover_candidate_pairs,
    verify_candidate_pairs,
)


def driver_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def jsonl_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(list(path.glob("*.jsonl")) + list(path.glob("*.json")))


def quota_for_file(total: int, file_idx: int, n_files: int) -> int:
    base = total // n_files
    remainder = total % n_files
    return base + (1 if file_idx < remainder else 0)


def load_balanced_jsonl(spark: SparkSession, input_path: str, size: int, text_col: str):
    frames = []
    files = jsonl_files(input_path)
    if not files:
        raise ValueError(f"No JSONL files found under {input_path}")

    for file_idx, json_path in enumerate(files):
        remaining = quota_for_file(size, file_idx, len(files))
        for pdf in pd.read_json(json_path, lines=True, chunksize=min(1000, max(1, remaining))):
            if remaining <= 0:
                break
            if len(pdf) > remaining:
                pdf = pdf.iloc[:remaining].copy()
            if "doc_id" not in pdf.columns or "parser_name" not in pdf.columns or text_col not in pdf.columns:
                raise ValueError(f"Expected doc_id, parser_name, {text_col} in {json_path}")
            pdf = pdf[["doc_id", "parser_name", text_col]].dropna(subset=[text_col]).copy()
            pdf["record_id"] = pdf["doc_id"].astype(str) + "::" + pdf["parser_name"].astype(str)
            pdf = pdf.rename(columns={text_col: "text"})
            frames.append(spark.createDataFrame(pdf[["record_id", "doc_id", "parser_name", "text"]]))
            remaining -= len(pdf)

    if not frames:
        return spark.createDataFrame(pd.DataFrame(columns=["record_id", "doc_id", "parser_name", "text"]))

    df = frames[0]
    for next_df in frames[1:]:
        df = df.unionByName(next_df)
    return df.repartition(min(max(4, size // 1000), 32))


def count_ground_truth_pairs(docs_df) -> int:
    rows = docs_df.groupBy("doc_id").agg(F.countDistinct("parser_name").alias("n")).where("n >= 2").collect()
    return int(sum((row.n * (row.n - 1)) // 2 for row in rows))


def evaluate_verified_pairs(verified_pairs_df, docs_df, total_truth: int) -> tuple[int, int, float, float, float]:
    mapping = docs_df.select(F.col("record_id").alias("rid"), F.col("doc_id").alias("base_doc_id"))
    left = mapping.select(F.col("rid").alias("doc_id_a"), F.col("base_doc_id").alias("base_a"))
    right = mapping.select(F.col("rid").alias("doc_id_b"), F.col("base_doc_id").alias("base_b"))
    labeled = verified_pairs_df.join(left, on="doc_id_a", how="inner").join(right, on="doc_id_b", how="inner")
    predicted = labeled.count()
    true_positive = labeled.where(F.col("base_a") == F.col("base_b")).count()
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / total_truth if total_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return int(predicted), int(true_positive), float(precision), float(recall), float(f1)


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_method(method: str, docs_df, band_rows, total_truth: int, args, out_root: Path, n_docs: int) -> dict:
    rss_before = driver_rss_gb()
    start = time.perf_counter()

    suspect_key_count = 0
    suspect_keys_size = 0
    full_band_index_size = 0
    if method == "full_minhash_lsh":
        full_band_index_size = write_df(
            band_rows,
            out_root / "outputs" / f"n_{n_docs}" / method / "band_index",
            True,
            args.local_io_mode,
        )
        pairs = recover_all_candidate_pairs(band_rows, max_bucket_size=args.max_bucket_size)
    elif method == "two_pass_recovery":
        suspect_keys = detect_suspect_keys(band_rows, num_partitions=args.suspect_partitions).dropDuplicates().persist()
        suspect_key_count = suspect_keys.count()
        suspect_keys_size = write_df(
            suspect_keys,
            out_root / "outputs" / f"n_{n_docs}" / method / "suspect_keys",
            True,
            args.local_io_mode,
        )
        pairs = recover_candidate_pairs(band_rows, suspect_keys, max_bucket_size=args.max_bucket_size)
        suspect_keys.unpersist()
    else:
        raise ValueError(method)

    pairs = pairs.persist()
    candidate_count = pairs.count()
    scored = verify_candidate_pairs(
        pairs_df=pairs,
        docs_df=docs_df,
        id_col="record_id",
        text_col="text",
        shingle_size=args.shingle_size,
        threshold=args.verify_threshold,
        max_shingles=args.max_shingles if args.max_shingles > 0 else None,
    ).persist()
    verified = scored.where("is_duplicate = true").persist()
    verified_count, true_positive, precision, recall, f1 = evaluate_verified_pairs(verified, docs_df, total_truth)

    output_dir = out_root / "outputs" / f"n_{n_docs}" / method
    candidate_size = write_df(pairs, output_dir / "candidate_pairs", True, args.local_io_mode)
    verified_size = write_df(verified, output_dir / "verified_pairs", True, args.local_io_mode)

    wall = time.perf_counter() - start
    rss_after = driver_rss_gb()

    pairs.unpersist()
    scored.unpersist()
    verified.unpersist()

    return {
        "n_docs": n_docs,
        "method": method,
        "wall_clock_sec": wall,
        "driver_rss_before_gb": rss_before,
        "driver_rss_after_gb": rss_after,
        "driver_rss_delta_gb": max(0.0, rss_after - rss_before),
        "band_rows": n_docs * args.num_bands,
        "suspect_keys": suspect_key_count,
        "candidate_pairs": candidate_count,
        "verified_pairs": verified_count,
        "true_positive_pairs": true_positive,
        "ground_truth_pairs": total_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "suspect_keys_size_bytes": suspect_keys_size,
        "full_band_index_size_bytes": full_band_index_size,
        "candidate_pairs_size_bytes": candidate_size,
        "verified_pairs_size_bytes": verified_size,
        "verify_threshold": args.verify_threshold,
        "num_perm": args.num_perm,
        "num_bands": args.num_bands,
        "shingle_size": args.shingle_size,
        "max_shingles": args.max_shingles,
        "notes": "ground truth: same doc_id across different parser_name values",
    }


def benchmark_size(spark, args, out_root: Path, size: int) -> list[dict]:
    docs = load_balanced_jsonl(spark, args.input_path, size=size, text_col=args.text_col).persist()
    n_docs = docs.count()
    total_truth = count_ground_truth_pairs(docs)

    band_rows = emit_band_rows(
        docs.select("record_id", "text"),
        id_col="record_id",
        text_col="text",
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        shingle_size=args.shingle_size,
        seed=args.seed,
        max_shingles=args.max_shingles if args.max_shingles > 0 else None,
    )
    if args.band_partitions > 0:
        band_rows = band_rows.repartition(args.band_partitions, "band_id", "band_key")
    band_rows = band_rows.persist()
    band_rows.count()

    rows = [
        benchmark_method("full_minhash_lsh", docs, band_rows, total_truth, args, out_root, n_docs),
        benchmark_method("two_pass_recovery", docs, band_rows, total_truth, args, out_root, n_docs),
    ]

    docs.unpersist()
    band_rows.unpersist()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full MinHashLSH recovery against two-pass LSHBloom recovery.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", default="spark_lshbloom/runs/two_pass_vs_full_lsh")
    parser.add_argument("--sizes", default="1000,5000")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bucket-size", type=int, default=10000)
    parser.add_argument("--verify-threshold", type=float, default=0.85)
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--band-partitions", type=int, default=32)
    parser.add_argument("--suspect-partitions", type=int, default=32)
    parser.add_argument("--local-io-mode", choices=["auto", "spark", "pyarrow", "python"], default="python")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--executor-memory", default="8g")
    parser.add_argument("--max-result-size", default="4g")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    reset_dir(out_root)
    (out_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")

    spark = make_spark(
        "spark-lshbloom-two-pass-vs-full-lsh",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
        arrow_enabled=not should_use_pyarrow_io(args.local_io_mode),
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
        max_result_size=args.max_result_size,
    )

    rows = []
    try:
        for size in parse_sizes(args.sizes):
            print(f"[two-pass-vs-full-lsh] n_docs={size:,}")
            rows.extend(benchmark_size(spark, args, out_root, size))
            write_results(out_root / "two_pass_vs_full_lsh_results.csv", rows)
    finally:
        spark.stop()

    write_results(out_root / "two_pass_vs_full_lsh_results.csv", rows)
    print("Wrote:", (out_root / "two_pass_vs_full_lsh_results.csv").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
