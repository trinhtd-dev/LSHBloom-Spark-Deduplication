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

# Sửa đường dẫn trỏ về đúng thư mục src/dedup/lsh (vì ta đã move code vào src)
LSH_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "dedup" / "lsh"
from datasketch import MinHash, MinHashLSH, MinHashLSHBloom, bBitMinHash
from datasketch.lsh import _optimal_param

TOKEN_RE = re.compile(r"\w+")

LSH_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "dedup" / "lsh"
sys.path.insert(0, str(LSH_DIR))
from lsh import LSHDeduper




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


def oph_signature_from_text(text: str, ngram_n: int, oph_bins: int):
    words = text.split()
    if not words:
        raise ValueError("Cannot build OPH signature for empty text")
    n_tokens = len(words)
    n_shingles = max(1, n_tokens - ngram_n + 1)
    sketch = OPHOptimalDenseSketch(num_bins=oph_bins)
    signature = sketch.compute(text, ngram=ngram_n)
    return signature, n_tokens, n_shingles


def minhash_lsh_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    workers: int,
    clear_filter_state: bool = False,
):
    """
    Full MinHash LSH benchmark path:
      1) tokenize / shingle
      2) build MinHash signature
      3) split into bands and compute band keys
      4) query candidate index
      5) verify candidates using MinHash Jaccard estimate
      6) insert non-duplicates

    This is the most complete MinHashLSH-style path in this benchmark.
    """
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    def band_keys_from_minhash(mh: MinHash) -> list[int]:
        if num_perm <= 0:
            return []
        b, r = _optimal_param(threshold, num_perm, 0.5, 0.5)
        keys: list[int] = []
        total = b * r
        if total <= 0:
            return keys
        for band in range(b):
            start = band * r
            end = start + r
            if end > len(mh.hashvalues):
                break
            digest = xxhash.xxh64(
                mh.hashvalues[start:end].tobytes() + band.to_bytes(4, "little"),
            ).intdigest()
            keys.append(digest)
        return keys

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()

    n_docs = 0
    total_tokens = 0
    total_shingles = 0
    docs_with_hits = 0
    total_hit_count = 0
    build_sig_sec = 0.0
    band_sec = 0.0
    query_sec = 0.0
    verify_sec = 0.0
    insert_sec = 0.0
    peak_ram_gb = process.memory_info().rss / (1024**3)

    band_index: dict[int, list[tuple[int, MinHash]]] = {}

    part_idx = 0
    start_time = time.perf_counter()
    print(f"[minhash_lsh] start threshold={threshold} num_perm={num_perm} workers={workers}")
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
            print(f"[minhash_lsh] loading part={part_idx:05d} rows={len(rows)}")
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            t0 = time.perf_counter()
            if pool is not None:
                payloads = pool.starmap(
                    minhash_payload_from_text,
                    [(t, num_perm, ngram_n, max_unique) for _, t in doc_texts],
                )
            else:
                payloads = [
                    minhash_payload_from_text(text, num_perm, ngram_n, max_unique)
                    for _, text in doc_texts
                ]
            build_sig_sec += time.perf_counter() - t0

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)

                t0 = time.perf_counter()
                keys = band_keys_from_minhash(mh)
                band_sec += time.perf_counter() - t0

                t0 = time.perf_counter()
                candidate_ids: set[int] = set()
                for key in keys:
                    for cand_id, _cand_mh in band_index.get(key, []):
                        if cand_id != doc_id:
                            candidate_ids.add(cand_id)
                query_sec += time.perf_counter() - t0

                is_dup = False
                if candidate_ids:
                    docs_with_hits += 1
                    total_hit_count += len(candidate_ids)
                    t0 = time.perf_counter()
                    for cand_id in candidate_ids:
                        cand_list = []
                        for key in keys:
                            cand_list.extend(band_index.get(key, []))
                        cand_mh = None
                        for cid, cmh in cand_list:
                            if cid == cand_id:
                                cand_mh = cmh
                                break
                        if cand_mh is None:
                            continue
                        if mh.jaccard(cand_mh) >= threshold:
                            is_dup = True
                            break
                    verify_sec += time.perf_counter() - t0

                if not is_dup:
                    t0 = time.perf_counter()
                    for key in keys:
                        band_index.setdefault(key, []).append((doc_id, mh))
                    insert_sec += time.perf_counter() - t0

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)
                total_tokens += n_tokens
                total_shingles += n_shingles
                n_docs += 1
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(sig_dir / f"doc_ids_{part_idx:05d}.npy", np.asarray(batch_doc_ids, dtype=np.int64))
            np.save(sig_dir / f"minhash_{part_idx:05d}.npy", np.vstack(batch_hashvalues).astype(np.uint64))

            if (part_idx + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[minhash_lsh] processed parts={part_idx + 1} docs={n_docs} "
                    f"build={build_sig_sec:.1f}s band={band_sec:.1f}s query={query_sec:.1f}s "
                    f"verify={verify_sec:.1f}s insert={insert_sec:.1f}s elapsed={elapsed:.1f}s"
                )

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed_total = time.perf_counter() - start_time
    print(f"[minhash_lsh] finished docs={n_docs} elapsed={elapsed_total:.1f}s")

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
        "band_sec": float(band_sec),
        "query_sec": float(query_sec),
        "verify_sec": float(verify_sec),
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
    start_time = time.perf_counter()
    print(f"[lsh_bbit] start threshold={threshold} num_perm={num_perm} workers={workers}")
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
            print(f"[lsh_bbit] reading part={part_idx:05d} rows={len(rows)}")
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

            if (part_idx + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[lsh_bbit] processed parts={part_idx + 1} docs={n_docs} "
                    f"build={build_sig_sec:.1f}s query={query_sec:.1f}s insert={insert_sec:.1f}s elapsed={elapsed:.1f}s"
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
    clear_filter_state: bool = False,
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


def _make_bloom_like_runner(algo_name: str, filter_class_importer):
    """
    Factory helper to create a runner for Blocked Bloom / BlowChoc.
    Both follow identical logic to lsh_bloom_runner, just with a different filter class.
    """
    def runner(
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
        clear_filter_state: bool = False,
    ):
        def get_text_from_row(row) -> str:
            if hasattr(row, "text_light_clean"):
                return row.text_light_clean
            if hasattr(row, "text"):
                return row.text
            raise AttributeError("Row has no text or text_light_clean column")

        sig_dir = out_dir / "signatures"
        sig_dir.mkdir(parents=True, exist_ok=True)

        LSHClass = filter_class_importer()
        lsh = LSHClass(threshold=threshold, num_perm=num_perm, fp=bloom_fp, n=bloom_n, save_dir=str(out_dir))
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
        start_time = time.perf_counter()
        print(f"[{algo_name}] start threshold={threshold} num_perm={num_perm}")
        pool = Pool(processes=workers) if workers and workers > 1 else None
        try:
            for table in iter_text_tables(
                subset_dir,
                batch_size=batch_rows,
                columns=["doc_id", "text_light_clean", "text"],
            ):
                print(f"[{algo_name}] loading part={part_idx:05d} rows={table.num_rows}")
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

                np.save(sig_dir / f"doc_ids_{part_idx:05d}.npy", np.asarray(batch_doc_ids, dtype=np.int64))
                np.save(sig_dir / f"minhash_{part_idx:05d}.npy", np.vstack(batch_hashvalues).astype(np.uint64))

                if (part_idx + 1) % 5 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[{algo_name}] processed parts={part_idx + 1} docs={n_docs} "
                        f"build={build_sig_sec:.1f}s query={query_sec:.1f}s insert={insert_sec:.1f}s "
                        f"elapsed={elapsed:.1f}s"
                    )

                part_idx += 1
                del df, batch_doc_ids, batch_hashvalues, rows, doc_texts
                gc.collect()
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        elapsed_total = time.perf_counter() - start_time
        print(f"[{algo_name}] finished docs={n_docs} elapsed={elapsed_total:.1f}s")

        return {
            "runner_type": algo_name,
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
    return runner


def _import_blocked_bloom():
    from lsh_blocked_bloom import MinHashLSHBlockedBloom
    return MinHashLSHBlockedBloom


def _import_blowchoc():
    from lsh_blowchoc import MinHashLSHBlowChoc
    return MinHashLSHBlowChoc


lsh_blocked_bloom_runner = _make_bloom_like_runner("LSH_Blocked_Bloom", _import_blocked_bloom)
lsh_blowchoc_runner = _make_bloom_like_runner("LSH_BlowChoc", _import_blowchoc)


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


def lsh_xor_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    fingerprint_bits: int,
    rebuild_every: int,
    force_compute_minhash: bool,
    workers: int,
    cache_minhash: bool,
    clear_filter_state: bool = False,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    minhash_root = out_dir / "minhashes" / str(num_perm)
    minhash_root.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"xor_filter_{threshold:.2f}_{num_perm}"
    save_dir.mkdir(parents=True, exist_ok=True)

    deduper = LSHXorDeduper(
        n=0,
        sim_threshold=threshold,
        num_perm=num_perm,
        minhash_root=str(minhash_root),
        save_dir=str(save_dir),
        recompute_minhashes=force_compute_minhash,
        ngram=ngram_n,
        fingerprint_bits=fingerprint_bits,
        rebuild_every=rebuild_every,
        cache_minhash=cache_minhash,
    )
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
    start_time = time.perf_counter()
    print(f"[lsh_xor] start threshold={threshold} num_perm={num_perm}")
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

            row_items = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in row_items]

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
                    mh, n_tokens, n_shingles = build_minhash_from_text(
                        text, num_perm=num_perm, ngram_n=ngram_n, max_unique=max_unique
                    )
                    build_sig_sec += time.perf_counter() - t0
                    payloads.append((mh.hashvalues.astype(np.uint64), n_tokens, n_shingles))

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)
                is_dup, q_sec, i_sec = deduper.deduplicate_minhash(mh)
                query_sec += q_sec
                insert_sec += i_sec

                if is_dup:
                    docs_with_hits += 1
                    total_hit_count += 1

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

            if (part_idx + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[lsh_xor] processed {n_docs} docs in {elapsed:.1f}s")

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, row_items, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    deduper.teardown()
    elapsed_total = time.perf_counter() - start_time
    print(f"[lsh_xor] finished {n_docs} docs in {elapsed_total:.1f}s")

    return {
        "runner_type": "LSH_XOR",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "xor_fingerprint_bits": int(fingerprint_bits),
        "xor_rebuild_every": int(rebuild_every),
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


def lsh_ribbon_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    fingerprint_bits: int,
    rebuild_every: int,
    force_compute_minhash: bool,
    workers: int,
    cache_minhash: bool,
    clear_filter_state: bool = False,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    minhash_root = out_dir / "minhashes" / str(num_perm)
    minhash_root.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"ribbon_filter_{threshold:.2f}_{num_perm}"
    save_dir.mkdir(parents=True, exist_ok=True)

    deduper = LSHRibbonDeduper(
        n=0,
        sim_threshold=threshold,
        num_perm=num_perm,
        minhash_root=str(minhash_root),
        save_dir=str(save_dir),
        recompute_minhashes=force_compute_minhash,
        ngram=ngram_n,
        fingerprint_bits=fingerprint_bits,
        rebuild_every=rebuild_every,
        cache_minhash=cache_minhash,
    )
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
    start_time = time.perf_counter()
    print(f"[lsh_ribbon] start threshold={threshold} num_perm={num_perm}")
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

            row_items = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in row_items]

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
                    mh, n_tokens, n_shingles = build_minhash_from_text(
                        text, num_perm=num_perm, ngram_n=ngram_n, max_unique=max_unique
                    )
                    build_sig_sec += time.perf_counter() - t0
                    payloads.append((mh.hashvalues.astype(np.uint64), n_tokens, n_shingles))

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)
                is_dup, q_sec, i_sec = deduper.deduplicate_minhash(mh)
                query_sec += q_sec
                insert_sec += i_sec

                if is_dup:
                    docs_with_hits += 1
                    total_hit_count += 1

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)

                n_docs += 1
                total_tokens += n_tokens
                total_shingles += n_shingles
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(sig_dir / f"doc_ids_{part_idx:05d}.npy", np.asarray(batch_doc_ids, dtype=np.int64))
            np.save(sig_dir / f"minhash_{part_idx:05d}.npy", np.vstack(batch_hashvalues).astype(np.uint64))

            if (part_idx + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[lsh_ribbon] processed {n_docs} docs in {elapsed:.1f}s")

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, row_items, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    deduper.teardown()
    elapsed_total = time.perf_counter() - start_time
    print(f"[lsh_ribbon] finished {n_docs} docs in {elapsed_total:.1f}s")

    return {
        "runner_type": "LSH_Ribbon",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "ribbon_fingerprint_bits": int(fingerprint_bits),
        "ribbon_rebuild_every": int(rebuild_every),
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


def lsh_zor_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    fingerprint_bits: int,
    force_compute_minhash: bool,
    workers: int,
    cache_minhash: bool,
    clear_filter_state: bool = False,
):
    def get_text_from_row(row) -> str:
        if hasattr(row, "text_light_clean"):
            return row.text_light_clean
        if hasattr(row, "text"):
            return row.text
        raise AttributeError("Row has no text or text_light_clean column")

    sig_dir = out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    minhash_root = out_dir / "minhashes" / str(num_perm)
    minhash_root.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"zor_filter_{threshold:.2f}_{num_perm}"
    save_dir.mkdir(parents=True, exist_ok=True)

    deduper = LSHZorDeduper(
        n=0,
        sim_threshold=threshold,
        num_perm=num_perm,
        minhash_root=str(minhash_root),
        save_dir=str(save_dir),
        recompute_minhashes=force_compute_minhash,
        ngram=ngram_n,
        fingerprint_bits=fingerprint_bits,
        rebuild_every=5000,
        cache_minhash=cache_minhash,
    )
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
    start_time = time.perf_counter()
    print(f"[lsh_zor] start threshold={threshold} num_perm={num_perm}")
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

            row_items = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in row_items]

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
                    mh, n_tokens, n_shingles = build_minhash_from_text(
                        text, num_perm=num_perm, ngram_n=ngram_n, max_unique=max_unique
                    )
                    build_sig_sec += time.perf_counter() - t0
                    payloads.append((mh.hashvalues.astype(np.uint64), n_tokens, n_shingles))

            for (doc_id, _), (hashvalues, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = build_minhash_from_hashvalues(hashvalues, num_perm)
                is_dup, q_sec, i_sec = deduper.deduplicate_minhash(mh)
                query_sec += q_sec
                insert_sec += i_sec

                if is_dup:
                    docs_with_hits += 1
                    total_hit_count += 1

                batch_doc_ids.append(doc_id)
                batch_hashvalues.append(hashvalues)

                n_docs += 1
                total_tokens += n_tokens
                total_shingles += n_shingles
                peak_ram_gb = max(peak_ram_gb, process.memory_info().rss / (1024**3))

            np.save(sig_dir / f"doc_ids_{part_idx:05d}.npy", np.asarray(batch_doc_ids, dtype=np.int64))
            np.save(sig_dir / f"minhash_{part_idx:05d}.npy", np.vstack(batch_hashvalues).astype(np.uint64))

            if (part_idx + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[lsh_zor] processed {n_docs} docs in {elapsed:.1f}s")

            part_idx += 1
            del df, batch_doc_ids, batch_hashvalues, row_items, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    deduper.teardown()
    elapsed_total = time.perf_counter() - start_time
    print(f"[lsh_zor] finished {n_docs} docs in {elapsed_total:.1f}s")

    return {
        "runner_type": "LSH_ZOR",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "zor_fingerprint_bits": int(fingerprint_bits),
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


def lsh_oph_doph_runner(
    subset_dir: Path,
    out_dir: Path,
    num_perm: int,
    threshold: float,
    ngram_n: int,
    max_unique: int,
    batch_rows: int,
    oph_bins: int,
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

    if num_perm != oph_bins:
        print(
            f"[warn] num_perm ({num_perm}) != oph_bins ({oph_bins}). Using oph_bins for OPH LSH."
        )
    num_perm = int(oph_bins)

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
            batch_signatures = []

            rows = list(df.itertuples(index=False))
            doc_texts = [(int(r.doc_id), get_text_from_row(r)) for r in rows]

            if pool is not None:
                t0 = time.perf_counter()
                payloads = pool.starmap(
                    oph_signature_from_text,
                    [(t, ngram_n, num_perm) for _, t in doc_texts],
                )
                build_sig_sec += time.perf_counter() - t0
            else:
                payloads = []
                for _, text in doc_texts:
                    t0 = time.perf_counter()
                    payloads.append(oph_signature_from_text(text, ngram_n, num_perm))
                    build_sig_sec += time.perf_counter() - t0

            for (doc_id, _), (signature, n_tokens, n_shingles) in zip(doc_texts, payloads):
                mh = sketch_to_minhash(signature, num_perm)

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
                batch_signatures.append(signature.astype(np.uint64))

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
                np.vstack(batch_signatures).astype(np.uint64),
            )

            part_idx += 1
            del df, batch_doc_ids, batch_signatures, rows, doc_texts
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "runner_type": "LSH_OPH_DOPH",
        "minhash_num_perm": int(num_perm),
        "minhash_threshold": float(threshold),
        "ngram_n": int(ngram_n),
        "max_unique_shingles": int(max_unique),
        "oph_bins": int(oph_bins),
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
    elif algo_name == "lsh_blocked_bloom":
        runner_result = lsh_blocked_bloom_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_blowchoc":
        runner_result = lsh_blowchoc_runner(
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
    elif algo_name == "lsh_xor":
        runner_result = lsh_xor_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_ribbon":
        runner_result = lsh_ribbon_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_zor":
        runner_result = lsh_zor_runner(
            subset_dir=subset_dir,
            out_dir=run_dir,
            **runner_config,
        )
    elif algo_name == "lsh_oph_doph":
        runner_result = lsh_oph_doph_runner(
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
    parser.add_argument("--algos", default="minhash_lsh,lsh_bbit,lsh_bloom,lsh_blocked_bloom,lsh_blowchoc,lsh_xor,lsh_zor")
    parser.add_argument("--b-bit", type=int, default=1)
    parser.add_argument("--bloom-fp", type=float, default=1e-5)
    parser.add_argument("--bloom-n", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=8)
    parser.add_argument("--oph-bins", type=int, default=128)
    parser.add_argument("--xor-fingerprint-bits", type=int, default=8)
    parser.add_argument("--xor-rebuild-every", type=int, default=5000)
    parser.add_argument("--xor-force-compute-minhash", action="store_true")
    parser.add_argument("--xor-cache-minhash", action="store_true")
    parser.add_argument("--cold-start-filters", action="store_true", help="Clear filter state only for speed benchmarks.")

    args = parser.parse_args()

    total_start = time.perf_counter()

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
    subset_prep_start = time.perf_counter()
    for n_docs in scale_points:
        subset_dir = subsets_root / f"text_{n_docs // 1000:03d}k"
        print(f"[scale] building subset {subset_dir.name} ({n_docs} docs)")
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
    subset_prep_sec = time.perf_counter() - subset_prep_start

    subset_df = pd.DataFrame(subset_summaries)
    subset_df.to_csv(out_root / "subset_summary.csv", index=False)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    results = []
    algo_timings = {}
    for algo in algos:
        print(f"[scale] running {algo} on {len(scale_points)} subset(s)")
        algo_start = time.perf_counter()
        runner_config = {
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "ngram_n": args.ngram,
            "max_unique": args.max_unique_shingles,
            "batch_rows": args.batch_rows,
            "workers": args.workers,
            "clear_filter_state": args.cold_start_filters,
        }
        if algo == "lsh_bbit":
            runner_config["b_bit"] = args.b_bit
        if algo == "lsh_bloom":
            runner_config["bloom_fp"] = args.bloom_fp
            runner_config["bloom_n"] = args.bloom_n
        if algo in ("lsh_blocked_bloom", "lsh_blowchoc"):
            runner_config["bloom_fp"] = args.bloom_fp
            runner_config["bloom_n"] = args.bloom_n
        if algo == "lsh_multiprobe":
            runner_config["num_probes"] = args.num_probes
        if algo == "lsh_xor":
            runner_config["fingerprint_bits"] = args.xor_fingerprint_bits
            runner_config["rebuild_every"] = args.xor_rebuild_every
            runner_config["force_compute_minhash"] = args.xor_force_compute_minhash
            runner_config["cache_minhash"] = args.xor_cache_minhash
            runner_config["clear_filter_state"] = args.cold_start_filters
        if algo == "lsh_zor":
            runner_config["fingerprint_bits"] = args.xor_fingerprint_bits
            runner_config["force_compute_minhash"] = args.xor_force_compute_minhash
            runner_config["cache_minhash"] = args.xor_cache_minhash
        if algo == "lsh_oph_doph":
            runner_config["oph_bins"] = args.oph_bins

        for n_docs in scale_points:
            subset_dir = subsets_root / f"text_{n_docs // 1000:03d}k"
            print(f"[scale] {algo} -> {subset_dir.name}")
            metrics = benchmark_one_subset(
                subset_dir=subset_dir,
                algo_name=algo,
                bench_root=bench_root,
                runner_config=runner_config,
            )
            results.append(metrics)
        algo_timings[algo] = time.perf_counter() - algo_start
        print(f"[scale] finished {algo} in {algo_timings[algo]:.2f}s")

    df = pd.DataFrame(results).sort_values(["algo_name", "n_docs"])
    df["disk_usage_gb"] = df["disk_usage_bytes"] / (1024**3)
    df.to_csv(out_root / "scale_results.csv", index=False)
    print(
        df[
            [
                "algo_name",
                "subset_name",
                "n_docs",
                "wall_clock_sec",
                "build_signature_sec",
                "query_sec",
                "insert_sec",
                "disk_usage_gb",
                "peak_ram_gb",
            ]
        ]
    )
    total_wall_clock_sec = time.perf_counter() - total_start
    timing_payload = {
        "total_wall_clock_sec": total_wall_clock_sec,
        "subset_prep_sec": subset_prep_sec,
        "algo_wall_clock_sec": algo_timings,
    }
    with open(out_root / "run_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)
    print(f"Total wall-clock sec: {total_wall_clock_sec:.2f}")
    print(f"Subset prep sec: {subset_prep_sec:.2f}")
    print(f"Algo wall-clock sec: {algo_timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
