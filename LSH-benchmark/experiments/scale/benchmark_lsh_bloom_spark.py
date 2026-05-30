import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import psutil
import xxhash
from datasketch import MinHash
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

import sys
LSH_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "dedup" / "lsh"
sys.path.insert(0, str(LSH_DIR))

from datasketch import MinHashLSHBloom

TOKEN_RE = re.compile(r"\w+")


@dataclass
class TimingStats:
    build_signature_sec: float = 0.0
    query_sec: float = 0.0
    insert_sec: float = 0.0
    total_docs_seen: int = 0
    docs_with_hits: int = 0
    total_hit_count: int = 0
    peak_rss_gb: float = 0.0


class ProcessMemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = self.process.memory_info().rss

    def update(self):
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)

    @property
    def peak_rss_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Spark benchmark for LSH Bloom following paper semantics.")
    parser.add_argument("--input", required=True, help="Input parquet directory with part-*.parquet files")
    parser.add_argument("--output-dir", default="spark_bloom_runs", help="Output directory for metrics and artifacts")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--max-unique-shingles", type=int, default=1024)
    parser.add_argument("--bloom-fp", type=float, default=1e-5)
    parser.add_argument("--bloom-n", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Spark parallelism hint")
    parser.add_argument("--batch-size", type=int, default=2000, help="Records per batch for local driver processing")
    parser.add_argument("--app-name", default="LSHBloomBenchmark")
    parser.add_argument("--master", default=None, help="Optional Spark master URL, e.g. local[*]")
    return parser.parse_args()


def write_result_csv(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run_id(args: argparse.Namespace) -> str:
    return f"bloom_np{args.num_perm}_t{args.threshold}_ng{args.ngram}_bfp{args.bloom_fp}".replace(".", "p")


def fast_hash_32(x: bytes) -> int:
    return xxhash.xxh32_intdigest(x)


def tokenize_for_minhash(text: str):
    return TOKEN_RE.findall(text.lower())


def iter_hashed_shingles(tokens, n, max_unique):
    seen = set()
    if not tokens:
        return
    if len(tokens) < n:
        sh = " ".join(tokens).encode("utf-8", errors="ignore")
        yield xxhash.xxh64_digest(sh)
        return

    for i in range(len(tokens) - n + 1):
        sh = " ".join(tokens[i : i + n]).encode("utf-8", errors="ignore")
        h = xxhash.xxh64_digest(sh)
        if h in seen:
            continue
        seen.add(h)
        yield h
        if len(seen) >= max_unique:
            break


def build_minhash_from_text(text: str, num_perm: int, ngram_n: int, max_unique: int):
    tokens = tokenize_for_minhash(text)
    mh = MinHash(num_perm=num_perm, hashfunc=fast_hash_32)
    n_shingles = 0
    for sh in iter_hashed_shingles(tokens, n=ngram_n, max_unique=max_unique):
        mh.update(sh)
        n_shingles += 1
    if n_shingles == 0:
        mh.update(b"__EMPTY__")
        n_shingles = 1
    return mh, len(tokens), n_shingles


def iter_rows_from_parquet(spark: SparkSession, input_dir: str) -> Iterator[dict]:
    df = spark.read.parquet(input_dir)
    cols = [c for c in ["doc_id", "text_light_clean", "text"] if c in df.columns]
    if "doc_id" not in cols:
        raise ValueError("Input parquet must contain a doc_id column")
    if "text_light_clean" not in cols and "text" not in cols:
        raise ValueError("Input parquet must contain text_light_clean or text")
    for row in df.select(*cols).toLocalIterator():
        yield row.asDict(recursive=True)


class SparkBloomBenchmark:
    """Standalone benchmark wrapper for paper-style LSHBloom execution."""

    def __init__(self, threshold: float, num_perm: int, bloom_fp: float, bloom_n: int, save_dir: str):
        self.deduper = MinHashLSHBloom(
            threshold=threshold,
            num_perm=num_perm,
            fp=bloom_fp,
            n=bloom_n,
            save_dir=save_dir,
        )
        self.threshold = threshold
        self.num_perm = num_perm

    def build_index(self, doc_df: DataFrame) -> tuple[DataFrame, DataFrame]:
        tokenizer = RegexTokenizer(
            inputCol="text",
            outputCol="tokens",
            pattern="\\W+",
        )
        token_df = tokenizer.transform(doc_df)

        def _build_mh(text: str):
            mh, _, _ = build_minhash_from_text(text, self.num_perm, 5, 1024)
            return [int(x) for x in mh.hashvalues.tolist()]

        mh_udf = F.udf(_build_mh, T.ArrayType(T.LongType()))
        feature_df = token_df.withColumn("hashvalues", mh_udf(F.col("text")))
        index_df = feature_df.select("doc_id", "hashvalues")
        return feature_df, index_df

    def generate_candidates(self, rows: Iterable[dict], stats: TimingStats, mem: ProcessMemoryMonitor):
        results = []
        for row in rows:
            doc_id = int(row["doc_id"])
            text = row.get("text_light_clean") or row.get("text") or ""

            t0 = time.perf_counter()
            mh, _, _ = build_minhash_from_text(text, self.num_perm, 5, 1024)
            stats.build_signature_sec += time.perf_counter() - t0

            # Paper semantics: query first, then insert unique docs into the Bloom-backed filter.
            t0 = time.perf_counter()
            if hasattr(self.deduper, "query"):
                is_dup = self.deduper.query(mh)
            else:
                is_dup = self.deduper.deduplicate(text, doc_id)
            stats.query_sec += time.perf_counter() - t0

            stats.total_docs_seen += 1
            if is_dup:
                stats.docs_with_hits += 1
                stats.total_hit_count += 1
            else:
                t0 = time.perf_counter()
                if hasattr(self.deduper, "insert"):
                    self.deduper.insert(mh)
                else:
                    # Fallback for implementations that only expose a combined API.
                    self.deduper.deduplicate(text, doc_id)
                stats.insert_sec += time.perf_counter() - t0

            results.append({
                "doc_id": doc_id,
                "is_duplicate": bool(is_dup),
                "minhash_perm": self.num_perm,
                "threshold": self.threshold,
            })
            mem.update()

        return results


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    spark_builder = SparkSession.builder.appName(args.app_name)
    if args.master:
        spark_builder = spark_builder.master(args.master)
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    total_docs = spark.read.parquet(args.input).count()
    bloom_n = args.bloom_n if args.bloom_n and args.bloom_n > 0 else total_docs
    print(f"[lsh_bloom] total_docs={total_docs} bloom_n={bloom_n}")

    filter_dir = run_dir / "bloom_filter"
    filter_dir.mkdir(parents=True, exist_ok=True)

    benchmark = SparkBloomBenchmark(
        threshold=args.threshold,
        num_perm=args.num_perm,
        bloom_fp=args.bloom_fp,
        bloom_n=bloom_n,
        save_dir=str(filter_dir),
    )

    timing = TimingStats()
    mem = ProcessMemoryMonitor()
    start_wall = time.perf_counter()

    print(f"[lsh_bloom] start threshold={args.threshold} num_perm={args.num_perm} workers={args.workers}")
    rows = []
    batch: list[dict] = []
    batch_idx = 0

    for row in iter_rows_from_parquet(spark, args.input):
        batch.append(row)
        if len(batch) >= args.batch_size:
            rows.append(batch)
            batch = []
    if batch:
        rows.append(batch)

    for batch_rows in rows:
        print(f"[lsh_bloom] processing batch={batch_idx:05d} size={len(batch_rows)}")
        batch_result = benchmark.generate_candidates(batch_rows, timing, mem)
        batch_path = run_dir / f"candidates_{batch_idx:05d}.json"
        with batch_path.open("w", encoding="utf-8") as f:
            json.dump(batch_result, f, indent=2)

        batch_idx += 1
        if batch_idx % 5 == 0:
            elapsed = time.perf_counter() - start_wall
            print(
                f"[lsh_bloom] batches={batch_idx} docs={timing.total_docs_seen} "
                f"build={timing.build_signature_sec:.1f}s query={timing.query_sec:.1f}s "
                f"insert={timing.insert_sec:.1f}s elapsed={elapsed:.1f}s"
            )

    if hasattr(benchmark.deduper, "teardown"):
        benchmark.deduper.teardown()
    wall_clock_sec = time.perf_counter() - start_wall

    result = {
        "algo_name": "lsh_bloom",
        "run_id": run_id,
        "input": args.input,
        "threshold": args.threshold,
        "num_perm": args.num_perm,
        "ngram": args.ngram,
        "max_unique_shingles": args.max_unique_shingles,
        "bloom_fp": args.bloom_fp,
        "bloom_n": args.bloom_n,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "total_docs_seen": timing.total_docs_seen,
        "docs_with_hits": timing.docs_with_hits,
        "total_hit_count": timing.total_hit_count,
        "build_signature_sec": timing.build_signature_sec,
        "query_sec": timing.query_sec,
        "insert_sec": timing.insert_sec,
        "wall_clock_sec": wall_clock_sec,
        "peak_rss_gb": mem.peak_rss_gb,
        "filter_disk_bytes": sum(p.stat().st_size for p in filter_dir.rglob("*") if p.is_file()),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_result_csv(run_dir / "metrics.csv", [result])
    spark.stop()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
