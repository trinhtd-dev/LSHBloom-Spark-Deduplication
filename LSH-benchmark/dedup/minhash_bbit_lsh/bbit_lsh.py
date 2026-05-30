from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from hashlib import blake2b

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
DEDUP_DIR = CURRENT_DIR.parent
LSH_DIR = DEDUP_DIR / "lsh"
sys.path.insert(0, str(DEDUP_DIR.parent / "synthetic_benchmark"))
sys.path.insert(0, str(DEDUP_DIR))

from config import DATA_PATH, WORK_DIR  # noqa: E402
from dedup_parsing_harness import DedupHarness  # noqa: E402

try:
    from datasketch import MinHash  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(LSH_DIR / "datasketch"))
    from datasketch import MinHash  # type: ignore  # noqa: E402


DEFAULT_B_BITS = 8
VALID_B_BITS = {1, 2, 4, 8, 16, 32}
BAND_HASH_BYTES = 16


tracemalloc.start()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def make_shingles(text: str, shingle_size: int) -> Set[str]:
    words = tokenize(text)
    if not words:
        return set()
    if len(words) < shingle_size:
        return set(words)
    return {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def choose_lsh_params(num_perm: int, num_bands: Optional[int], rows_per_band: Optional[int]) -> Optional[Tuple[int, int]]:
    if num_bands is None and rows_per_band is None:
        return None
    if num_bands is None:
        if rows_per_band <= 0:
            raise ValueError("rows_per_band must be positive")
        num_bands = num_perm // rows_per_band
    if rows_per_band is None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        rows_per_band = num_perm // num_bands
    if num_bands <= 0 or rows_per_band <= 0:
        raise ValueError("num_bands and rows_per_band must be positive")
    if num_bands * rows_per_band > num_perm:
        raise ValueError("num_bands * rows_per_band must be <= num_perm")
    return int(num_bands), int(rows_per_band)


def dtype_for_b_bits(b_bits: int) -> np.dtype:
    if b_bits <= 8:
        return np.dtype(np.uint8)
    if b_bits <= 16:
        return np.dtype(np.uint16)
    if b_bits <= 32:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


class DictBandLSHIndex:
    """Small deterministic LSH band index for already-compressed b-bit signatures."""

    def __init__(self, num_perm: int, threshold: float, params: Optional[Tuple[int, int]], max_bucket_size: int = 10000) -> None:
        self.num_perm = int(num_perm)
        self.threshold = float(threshold)
        self.max_bucket_size = int(max_bucket_size)
        self.num_bands, self.rows_per_band = self._resolve_params(params)
        self.hashranges = [
            (i * self.rows_per_band, (i + 1) * self.rows_per_band)
            for i in range(self.num_bands)
        ]
        self.buckets: Dict[Tuple[int, bytes], List[str]] = {}
        self.entry_count = 0
        self.skipped_large_buckets = 0

    def _resolve_params(self, params: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if params is not None:
            return params
        best = None
        best_error = float("inf")
        for bands in range(1, self.num_perm + 1):
            rows = self.num_perm // bands
            if rows <= 0 or bands * rows > self.num_perm:
                continue
            approx_threshold = (1.0 / bands) ** (1.0 / rows)
            error = abs(approx_threshold - self.threshold)
            if error < best_error:
                best = (bands, rows)
                best_error = error
        if best is None:
            raise ValueError("could not resolve LSH params")
        return best

    @staticmethod
    def _band_hash(band: np.ndarray) -> bytes:
        return blake2b(band.tobytes(), digest_size=BAND_HASH_BYTES).digest()

    def query(self, signature: np.ndarray) -> List[str]:
        candidates: Set[str] = set()
        for band_id, (start, end) in enumerate(self.hashranges):
            key = (band_id, self._band_hash(signature[start:end]))
            bucket = self.buckets.get(key)
            if not bucket:
                continue
            if self.max_bucket_size > 0 and len(bucket) > self.max_bucket_size:
                self.skipped_large_buckets += 1
                continue
            candidates.update(bucket)
        return list(candidates)

    def insert(self, doc_id: str, signature: np.ndarray) -> None:
        for band_id, (start, end) in enumerate(self.hashranges):
            key = (band_id, self._band_hash(signature[start:end]))
            self.buckets.setdefault(key, []).append(doc_id)
            self.entry_count += 1

    @property
    def bucket_count(self) -> int:
        return len(self.buckets)


class BBitMinHashLSH:
    """
    MinHashLSH pipeline with one deliberate change:
    full MinHash signature values are masked to their low b bits before LSH banding.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        b_bits: int = DEFAULT_B_BITS,
        shingle_size: int = 1,
        num_bands: Optional[int] = None,
        rows_per_band: Optional[int] = None,
        max_bucket_size: int = 10000,
    ) -> None:
        if b_bits not in VALID_B_BITS:
            raise ValueError(f"b_bits must be one of {sorted(VALID_B_BITS)}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")

        self.threshold = float(threshold)
        self.num_perm = int(num_perm)
        self.b_bits = int(b_bits)
        self.bbit_dtype = dtype_for_b_bits(self.b_bits)
        self.shingle_size = int(shingle_size)
        self.params = choose_lsh_params(self.num_perm, num_bands, rows_per_band)

        self.lsh = DictBandLSHIndex(
            num_perm=self.num_perm,
            threshold=self.threshold,
            params=self.params,
            max_bucket_size=max_bucket_size,
        )

        self._bbit_signatures: Dict[str, np.ndarray] = {}
        self._shingles: Dict[str, Set[str]] = {}
        self._inserted_ids: Set[str] = set()
        self.stats = Counter()
        self.jaccard_histogram = Counter()
        self.started_at = time.perf_counter()

    @property
    def num_bands(self) -> int:
        return self.lsh.num_bands

    @property
    def rows_per_band(self) -> int:
        return self.lsh.rows_per_band

    def build_full_minhash(self, shingles: Iterable[str]) -> MinHash:
        # datasketch.MinHash uses a stable SHA-1 based hash function by default.
        # Do not replace this with Python hash(), which is randomized per process.
        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8", errors="ignore"))
        return mh

    def to_bbit_signature(self, full_minhash: MinHash) -> np.ndarray:
        """
        The only algorithmic difference from standard MinHashLSH:
        keep only the low b bits of each full MinHash value before LSH banding.
        """
        mask = np.uint64((1 << self.b_bits) - 1)
        return np.bitwise_and(full_minhash.hashvalues, mask).astype(self.bbit_dtype, copy=False)

    def prepare(self, doc_id: str, text: str) -> Tuple[Set[str], MinHash, np.ndarray]:
        shingles = make_shingles(text, self.shingle_size)
        if not shingles:
            raise ValueError(f"empty document after preprocessing: {doc_id}")
        full_minhash = self.build_full_minhash(shingles)
        bbit_signature = self.to_bbit_signature(full_minhash)
        return shingles, full_minhash, bbit_signature

    def query(self, text: str, doc_id: str = "__query__") -> List[Tuple[str, float]]:
        shingles, _, bbit_signature = self.prepare(doc_id, text)
        candidate_ids = [str(candidate_id) for candidate_id in self.lsh.query(bbit_signature)]
        self.stats["candidate_pairs"] += len(candidate_ids)
        self.stats["queries"] += 1

        verified: List[Tuple[str, float]] = []
        for candidate_id in candidate_ids:
            score = jaccard(shingles, self._shingles[candidate_id])
            bucket = round(score, 1)
            self.jaccard_histogram[bucket] += 1
            if score >= self.threshold:
                verified.append((candidate_id, score))
        return sorted(verified, key=lambda item: item[1], reverse=True)

    def insert_prepared(self, doc_id: str, shingles: Set[str], bbit_signature: np.ndarray) -> None:
        doc_id = str(doc_id)
        self.lsh.insert(doc_id, bbit_signature)
        self._bbit_signatures[doc_id] = bbit_signature
        self._shingles[doc_id] = shingles
        self._inserted_ids.add(doc_id)
        self.stats["inserted_docs"] += 1

    def insert(self, doc_id: str, text: str) -> None:
        shingles, _, bbit_signature = self.prepare(str(doc_id), text)
        self.insert_prepared(str(doc_id), shingles, bbit_signature)

    def query_and_insert(self, doc_id: str, text: str) -> Tuple[bool, List[Tuple[str, float]]]:
        doc_id = str(doc_id)
        t0 = time.perf_counter()
        shingles, _, bbit_signature = self.prepare(doc_id, text)
        self.stats["signature_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)

        t0 = time.perf_counter()
        candidate_ids = [str(candidate_id) for candidate_id in self.lsh.query(bbit_signature)]
        self.stats["query_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        self.stats["candidate_pairs"] += len(candidate_ids)
        self.stats["processed_docs"] += 1

        verified: List[Tuple[str, float]] = []
        for candidate_id in candidate_ids:
            score = jaccard(shingles, self._shingles[candidate_id])
            self.jaccard_histogram[round(score, 1)] += 1
            if score >= self.threshold:
                verified.append((candidate_id, score))

        is_duplicate = bool(verified)
        if is_duplicate:
            self.stats["predicted_duplicates"] += 1
        else:
            t0 = time.perf_counter()
            self.insert_prepared(doc_id, shingles, bbit_signature)
            self.stats["insert_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        return is_duplicate, sorted(verified, key=lambda item: item[1], reverse=True)

    def estimated_signature_bytes(self) -> Dict[str, int]:
        n = int(self.stats["inserted_docs"])
        full_signature_bytes = n * self.num_perm * 8
        bbit_signature_bytes = sum(sig.nbytes for sig in self._bbit_signatures.values())
        band_bytes = self.rows_per_band * self.bbit_dtype.itemsize
        index_bytes = self.lsh.entry_count * (BAND_HASH_BYTES + 8) + self.lsh.bucket_count * 72
        return {
            "full_signature_bytes": int(full_signature_bytes),
            "bbit_signature_bytes": int(bbit_signature_bytes),
            "estimated_index_bytes": int(index_bytes),
            "bbit_dtype_bytes": int(self.bbit_dtype.itemsize),
            "band_payload_bytes": int(band_bytes),
        }

    def summary(self) -> Dict[str, object]:
        elapsed = time.perf_counter() - self.started_at
        timings = {
            "signature_sec": self.stats["signature_sec_x1e6"] / 1_000_000,
            "query_sec": self.stats["query_sec_x1e6"] / 1_000_000,
            "insert_sec": self.stats["insert_sec_x1e6"] / 1_000_000,
            "runtime_sec": elapsed,
        }
        return {
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "b_bits": self.b_bits,
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
            "shingle_size": self.shingle_size,
            "processed_docs": int(self.stats["processed_docs"]),
            "inserted_docs": int(self.stats["inserted_docs"]),
            "predicted_duplicates": int(self.stats["predicted_duplicates"]),
            "candidate_pairs": int(self.stats["candidate_pairs"]),
            "index_bucket_count": int(self.lsh.bucket_count),
            "index_entry_count": int(self.lsh.entry_count),
            "skipped_large_buckets": int(self.lsh.skipped_large_buckets),
            "max_bucket_size": int(self.lsh.max_bucket_size),
            "peak_memory_mb": round(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 4),
            "jaccard_histogram": {str(k): int(v) for k, v in sorted(self.jaccard_histogram.items())},
            **timings,
            **self.estimated_signature_bytes(),
        }


class BBitMinHashLSHDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        b_bits: int,
        shingle_size: int,
        minhash_root: str,
        num_bands: Optional[int] = None,
        rows_per_band: Optional[int] = None,
        max_bucket_size: int = 10000,
        recompute_minhashes: bool = False,
    ) -> None:
        super().__init__("minhash_bbit_lsh")
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.index = BBitMinHashLSH(
            threshold=sim_threshold,
            num_perm=num_perm,
            b_bits=b_bits,
            shingle_size=shingle_size,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            max_bucket_size=max_bucket_size,
        )

    def _cache_path(self, doc_id: str) -> str:
        return os.path.join(self.minhash_root, f"{doc_id}.pkl")

    def _load_or_prepare(self, text: str, doc_id: str) -> Tuple[Set[str], MinHash, np.ndarray]:
        shingles = make_shingles(text, self.index.shingle_size)
        if not shingles:
            raise ValueError(f"empty document after preprocessing: {doc_id}")

        mh_pkl = self._cache_path(doc_id)
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                full_minhash = pickle.load(f)
            if not isinstance(full_minhash, MinHash):
                raise TypeError(f"failed to parse MinHash cache: {mh_pkl}")
        else:
            full_minhash = self.index.build_full_minhash(shingles)
            with open(mh_pkl, "wb") as f:
                pickle.dump(full_minhash, f)

        # b-bit is applied only here, after full MinHash creation and before LSH banding.
        bbit_signature = self.index.to_bbit_signature(full_minhash)
        return shingles, full_minhash, bbit_signature

    def deduplicate(self, text: str, id: int) -> bool:
        doc_id = str(id)
        t0 = time.perf_counter()
        shingles, _, bbit_signature = self._load_or_prepare(text, doc_id)
        self.index.stats["signature_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)

        t0 = time.perf_counter()
        candidate_ids = [str(candidate_id) for candidate_id in self.index.lsh.query(bbit_signature)]
        self.index.stats["query_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        self.index.stats["candidate_pairs"] += len(candidate_ids)
        self.index.stats["processed_docs"] += 1

        verified = []
        for candidate_id in candidate_ids:
            score = jaccard(shingles, self.index._shingles[candidate_id])
            self.index.jaccard_histogram[round(score, 1)] += 1
            if score >= self.index.threshold:
                verified.append((candidate_id, score))

        is_duplicate = bool(verified)
        if is_duplicate:
            self.index.stats["predicted_duplicates"] += 1
        else:
            t0 = time.perf_counter()
            self.index.insert_prepared(doc_id, shingles, bbit_signature)
            self.index.stats["insert_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        return is_duplicate

    def write_stats(self, stats_json: str, stats_csv: str) -> None:
        summary = self.index.summary()
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            flat_summary = {k: v for k, v in summary.items() if not isinstance(v, dict)}
            writer = csv.DictWriter(f, fieldnames=list(flat_summary.keys()))
            writer.writeheader()
            writer.writerow(flat_summary)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run b-bit MinHashLSH on a benchmark_dfs dataset.")
    parser.add_argument("--input", required=True, help="Benchmark tag, e.g. test_p_0.1")
    parser.add_argument("--sim-threshold", "--threshold", dest="threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--b-bits", "--b-bit", dest="b_bits", type=int, default=DEFAULT_B_BITS)
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--rows-per-band", type=int, default=None)
    parser.add_argument("--shingle-size", "--ngram", dest="shingle_size", type=int, default=1)
    parser.add_argument("--max-bucket-size", type=int, default=10000)
    parser.add_argument("--force-compute-minhash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")

    result_dir = os.path.join(WORK_DIR, benchmark_tag, "minhash_bbit_lsh_results")
    minhash_root = os.path.join(
        result_dir,
        "minhashes",
        f"perm{args.num_perm}_shingle{args.shingle_size}_b{args.b_bits}",
    )
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    run_tag = f"minhash_bbit_lsh_{args.threshold}_{args.num_perm}_b{args.b_bits}"
    output_file = os.path.join(result_dir, f"{run_tag}_preds.csv")
    result_file = os.path.join(result_dir, f"{run_tag}_score.csv")
    stats_json = os.path.join(result_dir, f"{run_tag}_stats.json")
    stats_csv = os.path.join(result_dir, f"{run_tag}_stats.csv")

    deduper = BBitMinHashLSHDeduper(
        sim_threshold=args.threshold,
        num_perm=args.num_perm,
        b_bits=args.b_bits,
        shingle_size=args.shingle_size,
        minhash_root=minhash_root,
        num_bands=args.num_bands,
        rows_per_band=args.rows_per_band,
        max_bucket_size=args.max_bucket_size,
        recompute_minhashes=args.force_compute_minhash,
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.write_stats(stats_json, stats_csv)
    deduper.score(output_file, benchmark_csv, result_file)

    print(f"Saved predictions: {output_file}")
    print(f"Saved score:       {result_file}")
    print(f"Saved stats:       {stats_json}")


if __name__ == "__main__":
    main()
