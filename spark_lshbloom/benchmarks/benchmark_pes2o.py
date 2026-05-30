from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import psutil
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from spark_lshbloom import SparkLSHBloom


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_sizes(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def driver_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def ensure_java_available() -> None:
    if shutil.which("java"):
        return
    if "JAVA_HOME" in os.environ:
        java_home = Path(os.environ["JAVA_HOME"])
        if (java_home / "bin" / "java.exe").exists() or (java_home / "bin" / "java").exists():
            os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
            return
    for root in (Path("C:/Program Files/Eclipse Adoptium"), Path("C:/Program Files/Java")):
        if not root.exists():
            continue
        candidates = sorted(root.glob("jdk-17*"), reverse=True) + sorted(root.glob("jdk*"), reverse=True)
        for java_home in candidates:
            java_exe = java_home / "bin" / "java.exe"
            if java_exe.exists():
                os.environ["JAVA_HOME"] = str(java_home)
                os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
                return
    raise RuntimeError(
        "Java is required by PySpark but was not found. Install JDK 17 and set JAVA_HOME. "
        "On Windows, one option is: winget install EclipseAdoptium.Temurin.17.JDK"
    )


def ensure_windows_hadoop_available() -> None:
    if platform.system().lower() != "windows":
        return
    candidates = []
    if "HADOOP_HOME" in os.environ:
        candidates.append(Path(os.environ["HADOOP_HOME"]))
    candidates.extend(
        [
            Path("C:/hadoop"),
            Path("C:/winutils"),
            Path("C:/tmp/hadoop"),
            Path.cwd() / "hadoop",
            Path.cwd() / "winutils",
        ]
    )
    for root in candidates:
        winutils = root / "bin" / "winutils.exe"
        if winutils.exists():
            os.environ["HADOOP_HOME"] = str(root)
            os.environ["hadoop.home.dir"] = str(root)
            os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError(
        "PySpark on Windows requires winutils.exe for local file access. "
        "Create C:\\hadoop\\bin, put winutils.exe there, then set "
        "HADOOP_HOME=C:\\hadoop and add C:\\hadoop\\bin to PATH."
    )


def make_spark(
    app_name: str,
    master: str,
    shuffle_partitions: int,
    arrow_enabled: bool = True,
    arrow_max_records_per_batch: int = 2048,
) -> SparkSession:
    ensure_java_available()
    ensure_windows_hadoop_available()
    python_exe = sys.executable
    package_path = str(PACKAGE_ROOT)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if package_path not in existing_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = package_path + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe
    spark = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", str(bool(arrow_enabled)).lower())
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_max_records_per_batch))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.python.worker.reuse", "true")
        .config("spark.pyspark.python", python_exe)
        .config("spark.pyspark.driver.python", python_exe)
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .config("spark.yarn.appMasterEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    # spark.sparkContext.addPyFile(str(PACKAGE_ROOT / "spark_lshbloom"))
    return spark


def should_use_pyarrow_io(mode: str) -> bool:
    if mode in {"pyarrow", "python"}:
        return True
    if mode == "spark":
        return False
    return platform.system().lower() == "windows"


def load_sample_with_pyarrow(spark: SparkSession, input_path: str, size: int, text_col: str, id_col: str):
    dataset = ds.dataset(input_path, format="parquet")
    available = set(dataset.schema.names)
    if text_col not in available:
        raise ValueError(f"Missing text column '{text_col}'. Available columns: {sorted(available)}")
    columns = [text_col]
    if id_col in available:
        columns.insert(0, id_col)

    batches = []
    remaining = size
    for batch in dataset.to_batches(columns=columns, batch_size=min(10_000, max(1, size))):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(pa.Table.from_batches([batch]))
        remaining -= batch.num_rows

    if not batches:
        pdf = pd.DataFrame(columns=[id_col, text_col])
    else:
        table = pa.concat_tables(batches)
        pdf = table.to_pandas()

    if id_col not in pdf.columns:
        pdf.insert(0, id_col, range(len(pdf)))
    pdf = pdf[[id_col, text_col]].dropna(subset=[text_col]).head(size)
    return spark.createDataFrame(pdf).cache()


def load_sample(spark: SparkSession, input_path: str, size: int, text_col: str, id_col: str, io_mode: str):
    if should_use_pyarrow_io(io_mode):
        return load_sample_with_pyarrow(spark, input_path=input_path, size=size, text_col=text_col, id_col=id_col)
    df = spark.read.parquet(input_path)
    if id_col not in df.columns:
        df = df.withColumn(id_col, F.monotonically_increasing_id())
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Available columns: {df.columns}")
    return df.select(id_col, text_col).where(F.col(text_col).isNotNull()).limit(size).cache()


def write_banding_index(lsh: SparkLSHBloom, df, text_col: str, id_col: str, out_path: Path, repartition: int, io_mode: str):
    band_df = lsh._band_keys_column(df, text_col=text_col).select(
        F.col(id_col).alias("doc_id"),
        F.explode("band_keys").alias("band_key"),
    )
    if repartition > 0:
        band_df = band_df.repartition(repartition, "band_key")
    band_count = band_df.count()
    if should_use_pyarrow_io(io_mode):
        pdf = band_df.toPandas()
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        pq.write_table(table, out_path / "band_index.parquet", compression="zstd")
    else:
        band_df.write.mode("overwrite").option("compression", "zstd").parquet(str(out_path))
    return band_count


def benchmark_size(
    spark: SparkSession,
    input_path: str,
    out_root: Path,
    size: int,
    args,
) -> list[dict]:
    df = load_sample(spark, input_path=input_path, size=size, text_col=args.text_col, id_col=args.id_col, io_mode=args.local_io_mode)
    input_rows = df.count()
    results = []

    bloom_dir = out_root / "indexes" / f"lshbloom_{size}"
    reset_dir(bloom_dir)
    lshbloom = SparkLSHBloom(
        threshold=args.threshold,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        fp_prob=args.fp_prob,
        shingle_size=args.shingle_size,
        mode="lshbloom",
        expected_items=max(input_rows, 1),
        seed=args.seed,
        deduplicate_chunks=args.deduplicate_chunks,
        use_pandas_udf=args.local_io_mode != "python",
        max_shingles=args.max_shingles if args.max_shingles > 0 else None,
    )

    rss_before = driver_rss_gb()
    cpu_before = psutil.Process().cpu_times()
    start = time.perf_counter()
    lshbloom.fit(df, text_col=args.text_col, id_col=args.id_col)
    lshbloom.save(bloom_dir)
    wall = time.perf_counter() - start
    cpu_after = psutil.Process().cpu_times()
    rss_after = driver_rss_gb()
    bloom_meta = lshbloom.bloom.metadata() if lshbloom.bloom is not None else {}
    results.append(
        {
            "n_docs": input_rows,
            "method": "lshbloom",
            "wall_clock_sec": wall,
            "index_size_bytes": dir_size_bytes(bloom_dir),
            "index_size_gb": dir_size_bytes(bloom_dir) / (1024**3),
            "driver_rss_before_gb": rss_before,
            "driver_rss_after_gb": rss_after,
            "driver_rss_delta_gb": max(0.0, rss_after - rss_before),
            "driver_cpu_time_sec": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
            "num_perm": args.num_perm,
            "num_bands": lshbloom.config.resolved_num_bands,
            "rows_per_band": lshbloom.config.rows_per_band,
            "fp_prob": args.fp_prob,
            "shingle_size": args.shingle_size,
            "max_shingles": args.max_shingles,
            "band_entries": input_rows * lshbloom.config.resolved_num_bands,
            "bloom_num_bits": bloom_meta.get("num_bits", 0),
            "bloom_num_hashes": bloom_meta.get("num_hashes", 0),
            "notes": "mergeable bloom filter index",
        }
    )

    banding_dir = out_root / "indexes" / f"minhash_lsh_{size}"
    reset_dir(banding_dir)
    minhash_lsh = SparkLSHBloom(
        threshold=args.threshold,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        fp_prob=args.fp_prob,
        shingle_size=args.shingle_size,
        mode="minhash_lsh",
        expected_items=max(input_rows, 1),
        seed=args.seed,
        use_pandas_udf=args.local_io_mode != "python",
        max_shingles=args.max_shingles if args.max_shingles > 0 else None,
    )

    rss_before = driver_rss_gb()
    cpu_before = psutil.Process().cpu_times()
    start = time.perf_counter()
    band_entries = write_banding_index(
        minhash_lsh,
        df=df,
        text_col=args.text_col,
        id_col=args.id_col,
        out_path=banding_dir,
        repartition=args.index_partitions,
        io_mode=args.local_io_mode,
    )
    wall = time.perf_counter() - start
    cpu_after = psutil.Process().cpu_times()
    rss_after = driver_rss_gb()
    results.append(
        {
            "n_docs": input_rows,
            "method": "minhash_lsh",
            "wall_clock_sec": wall,
            "index_size_bytes": dir_size_bytes(banding_dir),
            "index_size_gb": dir_size_bytes(banding_dir) / (1024**3),
            "driver_rss_before_gb": rss_before,
            "driver_rss_after_gb": rss_after,
            "driver_rss_delta_gb": max(0.0, rss_after - rss_before),
            "driver_cpu_time_sec": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
            "num_perm": args.num_perm,
            "num_bands": minhash_lsh.config.resolved_num_bands,
            "rows_per_band": minhash_lsh.config.rows_per_band,
            "fp_prob": args.fp_prob,
            "shingle_size": args.shingle_size,
            "max_shingles": args.max_shingles,
            "band_entries": band_entries,
            "bloom_num_bits": "",
            "bloom_num_hashes": "",
            "notes": "explicit distributed band table parquet",
        }
    )

    df.unpersist()
    return results


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SparkLSHBloom vs distributed MinHashLSH band index on peS2o parquet.")
    parser.add_argument("--input-path", required=True, help="Parquet directory containing peS2o rows.")
    parser.add_argument("--out-dir", default="spark_lshbloom/runs/pes2o_benchmark")
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--id-col", default="doc_id")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=256)
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--fp-prob", type=float, default=0.001)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=2048, help="Maximum unique character shingles per document. Use 0 for unlimited.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate-chunks", type=int, default=16)
    parser.add_argument("--master", default="local[8]")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--index-partitions", type=int, default=64)
    parser.add_argument("--arrow-max-records-per-batch", type=int, default=2048)
    parser.add_argument(
        "--local-io-mode",
        choices=["auto", "spark", "pyarrow", "python"],
        default="auto",
        help="Use pyarrow local file I/O on Windows. 'python' also disables Arrow/Pandas UDF for stability.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")

    spark = make_spark(
        "spark-lshbloom-pes2o-benchmark",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
        arrow_enabled=args.local_io_mode != "python",
        arrow_max_records_per_batch=args.arrow_max_records_per_batch,
    )
    rows = []
    try:
        for size in parse_sizes(args.sizes):
            print(f"[benchmark] n_docs={size:,}")
            rows.extend(benchmark_size(spark, input_path=args.input_path, out_root=out_root, size=size, args=args))
            write_results(out_root / "benchmark_results.csv", rows)
    finally:
        spark.stop()

    write_results(out_root / "benchmark_results.csv", rows)
    print("Wrote:", (out_root / "benchmark_results.csv").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
