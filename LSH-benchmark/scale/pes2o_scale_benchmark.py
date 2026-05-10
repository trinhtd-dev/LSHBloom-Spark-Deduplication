import argparse
import gc
import json
import math
import re
import time
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import xxhash
from datasketch import MinHash, MinHashLSH, MinHashLSHBloom, bBitMinHash

TOKEN_RE = re.compile(r"\w+")

LSH_DIR = Path(__file__).resolve().parent.parent / "dedup" / "lsh"
sys.path.insert(0, str(LSH_DIR))
from lsh_multiprobe import MultiProbeLSH


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def count_rows_in_parquet_dir(parquet_dir: Path) -> int:
    total = 0
    for fp in sorted(parquet_dir.glob("part-*.parquet")):
        pf = pq.ParquetFile(fp)
        total += pf.metadata.num_rows
    return total


def reset_dir(path: Path):
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def iter_text_tables(parquet_dir: Path, batch_size=10_000, columns=None):
    for fp in sorted(parquet_dir.glob("part-*.parquet")):
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield pa.Table.from_batches([rb])


def create_text_subset_from_master(
    master_dir: Path,
    out_dir: Path,
    target_rows: int,
    rows_per_file: int = 10_000,
    columns=None,
):
    reset_dir(out_dir)
    written = 0
    file_idx = 0
    buffer_tables = []
    buffer_rows = 0

    for table in iter_text_tables(master_dir, batch_size=rows_per_file, columns=columns):
        remain = target_rows - written
        if remain <= 0:
            break
        if table.num_rows > remain:
            table = table.slice(0, remain)
        if table.num_rows == 0:
            continue

        buffer_tables.append(table)
        buffer_rows += table.num_rows
        written += table.num_rows

        while buffer_rows >= rows_per_file:
            merged = pa.concat_tables(buffer_tables)
            to_write = merged.slice(0, rows_per_file)
            out_path = out_dir / f"part-{file_idx:05d}.parquet"
            pq.write_table(to_write, out_path, compression="zstd")

            leftover = merged.slice(rows_per_file)
            buffer_tables = [leftover] if leftover.num_rows > 0 else []
            buffer_rows = leftover.num_rows
            file_idx += 1

        if written >= target_rows:
            break

    if buffer_rows > 0:
        merged = pa.concat_tables(buffer_tables)
        out_path = out_dir / f"part-{file_idx:05d}.parquet"
        pq.write_table(merged, out_path, compression="zstd")

    return {
        "target_rows": target_rows,
        "actual_rows": count_rows_in_parquet_dir(out_dir),
        "size_bytes": dir_size_bytes(out_dir),
        "num_files": len(list(out_dir.glob("part-*.parquet"))),
    }


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


def build_minhash_from_text(text, num_perm, ngram_n, max_unique):
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


def minhash_payload_from_text(text, num_perm, ngram_n, max_unique):
    mh, n_tokens, n_shingles = build_minhash_from_text(
        text, num_perm=num_perm, ngram_n=ngram_n, max_unique=max_unique
    )
    return mh.hashvalues.astype(np.uint64), n_tokens, n_shingles


def build_minhash_from_hashvalues(hashvalues, num_perm):
    mh = MinHash(num_perm=num_perm, hashfunc=fast_hash_32)
    mh.hashvalues = hashvalues
    return mh


def minhash_lsh_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    workers: int,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    process = psutil.Process()

    n_docs = 0
    total_tokens = 0
    total_shingles = 0

    docs_with_hits = 0
    total_hit_count = 0

    build_sig_sec = 0.0
    query_sec = 0.0
    insert_sec = 0.0

    peak_ram_gb = process.memory_info().rss / (1024**3)

    part_idx = 0
    pool = Pool(processes=workers) if workers and workers > 1 else None
    try:
        for table in iter_text_tables(
            subset_dir,
            batch_size=batch_rows,
            columns=["doc_id", "text_light_clean", "text"],
        ):
            df = table.to_pandas()
            batch_doc_ids = []
            batch_hashvalues = []

            rows = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            if pool is not None:
                t0 = time.perf_counter()
                payloads = pool.starmap(
                    minhash_payload_from_text,
                    [(t, num_perm, ngram_n, max_unique) for _, t in doc_texts],
                )
                build_sig_sec += time.perf_counter() - t0

                for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                    mh = build_minhash_from_hashvalues(hashvalues, num_perm)

                    t0 = time.perf_counter()
                    hits = lsh.query(mh)
                    query_sec += time.perf_counter() - t0

                    if hits:
                        docs_with_hits += 1
                        total_hit_count += len(hits)

                    t0 = time.perf_counter()
                    lsh.insert(str(doc_id), mh)
                    insert_sec += time.perf_counter() - t0

                    batch_doc_ids.append(doc_id)
                    batch_hashvalues.append(hashvalues)

                    n_docs += 1
                    total_tokens += n_tokens
                    total_shingles += n_shingles
                    peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))
            else:
                for row in rows:
                    doc_id = int(row.doc_id)
                    text = get_text_from_row(row)

                    t0 = time.perf_counter()
                    mh, n_tokens, n_shingles = build_minhash_from_text(
                        text, num_perm=num_perm, ngram_n=ngram_n, max_unique=max_unique
                    )
                    build_sig_sec += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    hits = lsh.query(mh)
                    query_sec += time.perf_counter() - t0

                    if hits:
                        docs_with_hits += 1
                        total_hit_count += len(hits)

                    t0 = time.perf_counter()
                    lsh.insert(str(doc_id), mh)
                    insert_sec += time.perf_counter() - t0

                    batch_doc_ids.append(doc_id)
                    batch_hashvalues.append(mh.hashvalues.astype(np.uint64))

                    n_docs += 1
                    total_tokens += n_tokens
                    total_shingles += n_shingles
                    peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(
                sig_dir / f"doc_ids_{part_idx:05d}.npy",
                np.asarray(batch_doc_ids, dtype=np.int64),
            )
            np.save(
                sig_dir / f"minhash_{part_idx:05d}.npy",
                np.vstack(batch_hashvalues).astype(np.uint64),
            )

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "runner_type": "MinHashLSH",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "total_docs_seen": int(n_docs),
        "total_tokens": int(total_tokens),
        "total_shingles_kept": int(total_shingles),
        "docs_with_hits": int(docs_with_hits),
        "total_hit_count": int(total_hit_count),
        "build_signature_sec": float(build_sig_sec),
        "query_sec": float(query_sec),
        "insert_sec": float(insert_sec),
        "peak_ram_gb": float(peak_ram_gb),
    }


def lsh_bbit_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    b_bit: int,
    workers: int,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    process = psutil.Process()

    bbit_store: dict[int, bBitMinHash] = {}

    n_docs = 0
    total_tokens = 0
    total_shingles = 0

    docs_with_hits = 0
    total_hit_count = 0

    build_sig_sec = 0.0
    query_sec = 0.0
    verify_sec = 0.0
    insert_sec = 0.0

    peak_ram_gb = process.memory_info().rss / (1024**3)

    part_idx = 0
    pool = Pool(processes=workers) if workers and workers > 1 else None
    try:
        for table in iter_text_tables(
            subset_dir,
            batch_size=batch_rows,
            columns=["doc_id", "text_light_clean", "text"],
        ):
            df = table.to_pandas()
            batch_doc_ids = []
            batch_hashvalues = []

            rows = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            if pool is not None:
                t0 = time.perf_counter()
                payloads = pool.starmap(
                    minhash_payload_from_text,
                    [(t, num_perm, ngram_n, max_unique) for _, t in doc_texts],
                )
                build_sig_sec += time.perf_counter() - t0
            else:
                payloads = []
                for _, text in doc_texts:
                    t0 = time.perf_counter()
                    payloads.append(minhash_payload_from_text(text, num_perm, ngram_n, max_unique))
                    build_sig_sec += time.perf_counter() - t0

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)

                t0 = time.perf_counter()
                hits = lsh.query(mh)
                query_sec += time.perf_counter() - t0

                is_dup = False
                if hits:
                    docs_with_hits += 1
                    total_hit_count += len(hits)
                    bb_query = bBitMinHash(mh, b=b_bit)
                    t0 = time.perf_counter()
                    for cand_id in hits:
                        bb_cand = bbit_store.get(int(cand_id))
                        if bb_cand is None:
                            continue
                        if bb_query.jaccard(bb_cand) >= threshold:
                            is_dup = True
                            break
                    verify_sec += time.perf_counter() - t0

                if not is_dup:
                    t0 = time.perf_counter()
                    lsh.insert(str(doc_id), mh)
                    insert_sec += time.perf_counter() - t0
                    bbit_store[doc_id] = bBitMinHash(mh, b=b_bit)

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)

                n_docs += 1
                total_tokens += n_tokens
                total_shingles += n_shingles
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(
                sig_dir / f"doc_ids_{part_idx:05d}.npy",
                np.asarray(batch_doc_ids, dtype=np.int64),
            )
            np.save(
                sig_dir / f"minhash_{part_idx:05d}.npy",
                np.vstack(batch_hashvalues).astype(np.uint64),
            )

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "runner_type": "LSH_BBit",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "b_bit": int(b_bit),
        "total_docs_seen": int(n_docs),
        "total_tokens": int(total_tokens),
        "total_shingles_kept": int(total_shingles),
        "docs_with_hits": int(docs_with_hits),
        "total_hit_count": int(total_hit_count),
        "build_signature_sec": float(build_sig_sec),
        "query_sec": float(query_sec),
        "verify_sec": float(verify_sec),
        "insert_sec": float(insert_sec),
        "peak_ram_gb": float(peak_ram_gb),
    }


def lsh_bloom_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    bloom_fp: float,
    bloom_n: int,
    workers: int,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSHBloom(threshold=threshold, num_perm=num_perm, fp=bloom_fp, n=bloom_n, save_dir=str(out_dir))
    process = psutil.Process()

    n_docs = 0
    total_tokens = 0
    total_shingles = 0

    docs_with_hits = 0
    total_hit_count = 0

    build_sig_sec = 0.0
    query_sec = 0.0
    insert_sec = 0.0

    peak_ram_gb = process.memory_info().rss / (1024**3)

    part_idx = 0
    pool = Pool(processes=workers) if workers and workers > 1 else None
    try:
        for table in iter_text_tables(
            subset_dir,
            batch_size=batch_rows,
            columns=["doc_id", "text_light_clean", "text"],
        ):
            df = table.to_pandas()
            batch_doc_ids = []
            batch_hashvalues = []

            rows = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            if pool is not None:
                t0 = time.perf_counter()
                payloads = pool.starmap(
                    minhash_payload_from_text,
                    [(t, num_perm, ngram_n, max_unique) for _, t in doc_texts],
                )
                build_sig_sec += time.perf_counter() - t0
            else:
                payloads = []
                for _, text in doc_texts:
                    t0 = time.perf_counter()
                    payloads.append(minhash_payload_from_text(text, num_perm, ngram_n, max_unique))
                    build_sig_sec += time.perf_counter() - t0

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)

                t0 = time.perf_counter()
                is_dup = lsh.query(mh)
                query_sec += time.perf_counter() - t0

                if is_dup:
                    docs_with_hits += 1
                    total_hit_count += 1
                else:
                    t0 = time.perf_counter()
                    lsh.insert(mh)
                    insert_sec += time.perf_counter() - t0

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)

                n_docs += 1
                total_tokens += n_tokens
                total_shingles += n_shingles
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(
                sig_dir / f"doc_ids_{part_idx:05d}.npy",
                np.asarray(batch_doc_ids, dtype=np.int64),
            )
            np.save(
                sig_dir / f"minhash_{part_idx:05d}.npy",
                np.vstack(batch_hashvalues).astype(np.uint64),
            )

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "runner_type": "LSH_Bloom",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "bloom_fp": float(bloom_fp),
        "bloom_n": int(bloom_n),
        "total_docs_seen": int(n_docs),
        "total_tokens": int(total_tokens),
        "total_shingles_kept": int(total_shingles),
        "docs_with_hits": int(docs_with_hits),
        "total_hit_count": int(total_hit_count),
        "build_signature_sec": float(build_sig_sec),
        "query_sec": float(query_sec),
        "insert_sec": float(insert_sec),
        "peak_ram_gb": float(peak_ram_gb),
    }


def lsh_multiprobe_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    num_probes: int,
    workers: int,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    lsh = MultiProbeLSH(threshold=threshold, num_perm=num_perm, num_probes=num_probes)
    process = psutil.Process()

    n_docs = 0
    total_tokens = 0
    total_shingles = 0

    docs_with_hits = 0
    total_hit_count = 0

    build_sig_sec = 0.0
    query_sec = 0.0
    insert_sec = 0.0

    peak_ram_gb = process.memory_info().rss / (1024**3)

    part_idx = 0
    pool = Pool(processes=workers) if workers and workers > 1 else None
    try:
        for table in iter_text_tables(
            subset_dir,
            batch_size=batch_rows,
            columns=["doc_id", "text_light_clean", "text"],
        ):
            df = table.to_pandas()
            batch_doc_ids = []
            batch_hashvalues = []

            rows = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            if pool is not None:
                t0 = time.perf_counter()
                payloads = pool.starmap(
                    minhash_payload_from_text,
                    [(t, num_perm, ngram_n, max_unique) for _, t in doc_texts],
                )
                build_sig_sec += time.perf_counter() - t0
            else:
                payloads = []
                for _, text in doc_texts:
                    t0 = time.perf_counter()
                    payloads.append(minhash_payload_from_text(text, num_perm, ngram_n, max_unique))
                    build_sig_sec += time.perf_counter() - t0

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)

                t0 = time.perf_counter()
                hits = lsh.query(mh)
                query_sec += time.perf_counter() - t0

                if hits:
                    docs_with_hits += 1
                    total_hit_count += len(hits)

                t0 = time.perf_counter()
                lsh.insert(str(doc_id), mh)
                insert_sec += time.perf_counter() - t0

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)

                n_docs += 1
                total_tokens += n_tokens
                total_shingles += n_shingles
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(
                sig_dir / f"doc_ids_{part_idx:05d}.npy",
                np.asarray(batch_doc_ids, dtype=np.int64),
            )
            np.save(
                sig_dir / f"minhash_{part_idx:05d}.npy",
                np.vstack(batch_hashvalues).astype(np.uint64),
            )

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "runner_type": "LSH_MultiProbe",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "num_probes": int(num_probes),
        "total_docs_seen": int(n_docs),
        "total_tokens": int(total_tokens),
        "total_shingles_kept": int(total_shingles),
        "docs_with_hits": int(docs_with_hits),
        "total_hit_count": int(total_hit_count),
        "build_signature_sec": float(build_sig_sec),
        "query_sec": float(query_sec),
        "insert_sec": float(insert_sec),
        "peak_ram_gb": float(peak_ram_gb),
    }


def benchmark_one_subset(
    subset_dir: Path,
    algo_name: str,
    bench_root: Path,
    runner_config: dict,
):
    run_dir = bench_root / f"{algo_name}__{subset_dir.name}"
    reset_dir(run_dir)

    t0 = time.perf_counter()
    if algo_name == "minhash_lsh":
        runner_result = minhash_lsh_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_bbit":
        runner_result = lsh_bbit_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_bloom":
        runner_result = lsh_bloom_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_multiprobe":
        runner_result = lsh_multiprobe_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    else:
        raise ValueError(f"Unknown algo_name: {algo_name}")
    wall_clock_sec = time.perf_counter() - t0

    metrics = {
        "algo_name": algo_name,
        "subset_name": subset_dir.name,
        "n_docs": count_rows_in_parquet_dir(subset_dir),
        "wall_clock_sec": wall_clock_sec,
        "disk_usage_bytes": dir_size_bytes(run_dir),
        "config": runner_config,
    }
    metrics.update(runner_result)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_scale_points(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale benchmark for MinHashLSH on peS2o subsets.")
    parser.add_argument("--text-parquet-dir", required=True, help="Directory with part-*.parquet files.")
    parser.add_argument("--out-dir", default=str(Path("scale_runs")), help="Output root directory.")
    parser.add_argument("--scale-points", default="200000,400000,600000,800000,1000000")
    parser.add_argument("--rows-per-subset-file", type=int, default=10_000)

    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--max-unique-shingles", type=int, default=1024)
    parser.add_argument("--batch-rows", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--algos", default="minhash_lsh,lsh_bbit,lsh_bloom")
    parser.add_argument("--b-bit", type=int, default=1)
    parser.add_argument("--bloom-fp", type=float, default=1e-5)
    parser.add_argument("--bloom-n", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=8)

    args = parser.parse_args()

    text_dir = Path(args.text_parquet_dir)
    out_root = Path(args.out_dir)

    subsets_root = out_root / "subsets"
    bench_root = out_root / "benchmarks"
    subsets_root.mkdir(parents=True, exist_ok=True)
    bench_root.mkdir(parents=True, exist_ok=True)

    scale_points = parse_scale_points(args.scale_points)
    text_columns = [
        "doc_id",
        "source",
        "text_light_clean",
        "text",
        "n_words",
        "n_chars",
        "exact_hash",
    ]

    subset_summaries = []
    for n_docs in scale_points:
        subset_dir = subsets_root / f"text_{n_docs // 1000:03d}k"
        summary = create_text_subset_from_master(
            master_dir=text_dir,
            out_dir=subset_dir,
            target_rows=n_docs,
            rows_per_file=args.rows_per_subset_file,
            columns=text_columns,
        )
        subset_summaries.append(
            {
                "subset_name": subset_dir.name,
                "n_docs": n_docs,
                "actual_rows": summary["actual_rows"],
                "num_files": summary["num_files"],
                "size_gb": round(summary["size_bytes"] / (1024**3), 4),
            }
        )

    subset_df = pd.DataFrame(subset_summaries)
    subset_df.to_csv(out_root / "subset_summary.csv", index=False)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    results = []
    for algo in algos:
        runner_config = {
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "ngram_n": args.ngram,
            "max_unique": args.max_unique_shingles,
            "batch_rows": args.batch_rows,
            "workers": args.workers,
        }
        if algo == "lsh_bbit":
            runner_config["b_bit"] = args.b_bit
        if algo == "lsh_bloom":
            runner_config["bloom_fp"] = args.bloom_fp
            runner_config["bloom_n"] = args.bloom_n
        if algo == "lsh_multiprobe":
            runner_config["num_probes"] = args.num_probes

        for n_docs in scale_points:
            subset_dir = subsets_root / f"text_{n_docs // 1000:03d}k"
            metrics = benchmark_one_subset(
                subset_dir=subset_dir,
                algo_name=algo,
                bench_root=bench_root,
                runner_config=runner_config,
            )
            results.append(metrics)

    df = pd.DataFrame(results).sort_values(["algo_name", "n_docs"])
    df["disk_usage_gb"] = df["disk_usage_bytes"] / (1024**3)
    df.to_csv(out_root / "scale_results.csv", index=False)
    print(df[["algo_name", "subset_name", "n_docs", "wall_clock_sec", "disk_usage_gb", "peak_ram_gb"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
