from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_pes2o import dir_size_bytes, load_sample, make_spark, parse_sizes, reset_dir
from benchmark_pes2o import should_use_pyarrow_io
import pyarrow as pa
import pyarrow.parquet as pq
from spark_lshbloom import detect_suspect_keys, emit_band_rows, recover_candidate_pairs, verify_candidate_pairs


def parse_record_id_cols(raw: str) -> list[str] | None:
    cols = [part.strip() for part in raw.split(",") if part.strip()]
    return cols or None


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_df(df, path: Path, enabled: bool, io_mode: str) -> int:
    if not enabled:
        return 0
    reset_dir(path)
    if should_use_pyarrow_io(io_mode):
        pdf = df.toPandas()
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        pq.write_table(table, path / "part-00000.parquet", compression="zstd")
    else:
        df.write.mode("overwrite").option("compression", "zstd").parquet(str(path))
    return dir_size_bytes(path)


def benchmark_size(spark, args, out_root: Path, size: int) -> dict:
    df = load_sample(
        spark,
        input_path=args.input_path,
        size=size,
        text_col=args.text_col,
        id_col=args.id_col,
        io_mode=args.local_io_mode,
        cache=size <= args.cache_threshold,
        record_id_cols=parse_record_id_cols(args.record_id_cols),
    )
    n_docs = df.count()

    t0 = time.perf_counter()
    band_rows = emit_band_rows(
        df,
        id_col=args.id_col,
        text_col=args.text_col,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        shingle_size=args.shingle_size,
        seed=args.seed,
        max_shingles=args.max_shingles if args.max_shingles > 0 else None,
    )
    if args.band_partitions > 0:
        band_rows = band_rows.repartition(args.band_partitions, "band_id", "band_key")
    band_rows = band_rows.persist()
    band_row_count = band_rows.count()
    emit_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    suspect_keys = detect_suspect_keys(band_rows, num_partitions=args.suspect_partitions).dropDuplicates()
    suspect_keys = suspect_keys.persist()
    suspect_key_count = suspect_keys.count()
    suspect_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    pairs = recover_candidate_pairs(
        band_rows_df=band_rows,
        suspect_keys_df=suspect_keys,
        max_bucket_size=args.max_bucket_size,
    )
    pair_count = pairs.count()
    pair_sec = time.perf_counter() - t2

    verify_sec = 0.0
    verified_count = 0
    scored_pairs_size = 0
    verified_pairs_size = 0
    scored_pairs = None
    verified_pairs = None
    if args.verify_pairs:
        t3 = time.perf_counter()
        scored_pairs = verify_candidate_pairs(
            pairs_df=pairs,
            docs_df=df,
            id_col=args.id_col,
            text_col=args.text_col,
            shingle_size=args.shingle_size,
            threshold=args.verify_threshold,
            max_shingles=args.max_shingles if args.max_shingles > 0 else None,
        ).persist()
        verified_pairs = scored_pairs.where("is_duplicate = true")
        verified_count = verified_pairs.count()
        verify_sec = time.perf_counter() - t3

    output_root = out_root / "outputs" / f"n_{n_docs}"
    band_rows_size = write_df(band_rows, output_root / "band_rows", args.write_intermediate, args.local_io_mode)
    suspect_keys_size = write_df(suspect_keys, output_root / "suspect_keys", True, args.local_io_mode)
    pairs_size = write_df(pairs, output_root / "candidate_pairs", True, args.local_io_mode)
    if args.verify_pairs:
        scored_pairs_size = write_df(scored_pairs, output_root / "scored_pairs", args.write_scored_pairs, args.local_io_mode)
        verified_pairs_size = write_df(verified_pairs, output_root / "verified_pairs", True, args.local_io_mode)

    if size <= args.cache_threshold:
        df.unpersist()
    band_rows.unpersist()
    suspect_keys.unpersist()
    if scored_pairs is not None:
        scored_pairs.unpersist()

    return {
        "n_docs": n_docs,
        "method": "two_pass_recovery",
        "wall_clock_sec": emit_sec + suspect_sec + pair_sec + verify_sec,
        "emit_band_rows_sec": emit_sec,
        "detect_suspect_keys_sec": suspect_sec,
        "recover_pairs_sec": pair_sec,
        "verify_pairs_sec": verify_sec,
        "band_rows": band_row_count,
        "suspect_keys": suspect_key_count,
        "candidate_pairs": pair_count,
        "verified_pairs": verified_count,
        "band_rows_size_bytes": band_rows_size,
        "suspect_keys_size_bytes": suspect_keys_size,
        "candidate_pairs_size_bytes": pairs_size,
        "scored_pairs_size_bytes": scored_pairs_size,
        "verified_pairs_size_bytes": verified_pairs_size,
        "verify_threshold": args.verify_threshold,
        "num_perm": args.num_perm,
        "num_bands": args.num_bands,
        "shingle_size": args.shingle_size,
        "max_shingles": args.max_shingles,
        "max_bucket_size": args.max_bucket_size,
        "notes": "two-pass suspect-key recovery; optional exact Jaccard verification on recovered pairs",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark two-pass candidate recovery for SparkLSHBloom.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", default="spark_lshbloom/runs/two_pass_recovery")
    parser.add_argument("--sizes", default="10000")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--id-col", default="doc_id")
    parser.add_argument(
        "--record-id-cols",
        default="",
        help="Comma-separated columns to concatenate into a unique id, e.g. doc_id,parser_name for multi-parser JSONL data.",
    )
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=32)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bucket-size", type=int, default=10000)
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--band-partitions", type=int, default=64)
    parser.add_argument("--suspect-partitions", type=int, default=64)
    parser.add_argument("--local-io-mode", choices=["auto", "spark", "pyarrow", "python"], default="auto")
    parser.add_argument("--cache-threshold", type=int, default=100_000)
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--executor-memory", default="8g")
    parser.add_argument("--max-result-size", default="4g")
    parser.add_argument("--write-intermediate", action="store_true", help="Also write full band rows. This can be large.")
    parser.add_argument("--verify-pairs", action="store_true", help="Verify recovered candidate pairs with exact character-shingle Jaccard.")
    parser.add_argument("--verify-threshold", type=float, default=0.85)
    parser.add_argument("--write-scored-pairs", action="store_true", help="Also write all scored candidate pairs, not only verified pairs.")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")

    spark = make_spark(
        "spark-lshbloom-two-pass-recovery",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
        arrow_enabled=args.local_io_mode != "python",
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
        max_result_size=args.max_result_size,
    )

    rows = []
    try:
        for size in parse_sizes(args.sizes):
            print(f"[two-pass] n_docs={size:,}")
            rows.append(benchmark_size(spark, args, out_root, size))
            write_results(out_root / "two_pass_results.csv", rows)
    finally:
        spark.stop()

    write_results(out_root / "two_pass_results.csv", rows)
    print("Wrote:", (out_root / "two_pass_results.csv").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
