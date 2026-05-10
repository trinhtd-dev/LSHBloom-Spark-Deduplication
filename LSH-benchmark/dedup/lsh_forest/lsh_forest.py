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

CURRENT_DIR = Path(__file__).resolve().parent
DEDUP_DIR = CURRENT_DIR.parent
LSH_DIR = DEDUP_DIR / "lsh"
sys.path.insert(0, str(DEDUP_DIR.parent / "synthetic_benchmark"))
sys.path.insert(0, str(DEDUP_DIR))

from config import DATA_PATH, WORK_DIR  # noqa: E402
from dedup_parsing_harness import DedupHarness  # noqa: E402

try:
    from datasketch import MinHash, MinHashLSHForest  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(LSH_DIR / "datasketch"))
    from datasketch import MinHash, MinHashLSHForest  # type: ignore  # noqa: E402


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


class MinHashLSHForestDeduper(DedupHarness):
    """
    Near-duplicate deduper using datasketch.MinHashLSHForest.

    Pipeline:
    text -> preprocessing -> shingling -> MinHash signature -> LSH Forest top-k
    candidates -> exact Jaccard verification.
    """

    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        shingle_size: int,
        minhash_root: str,
        num_trees: int = 8,
        top_k: int = 50,
        index_batch_size: int = 100,
        recompute_minhashes: bool = False,
    ) -> None:
        super().__init__("lsh_forest")
        if not 0.0 <= sim_threshold <= 1.0:
            raise ValueError("sim_threshold must be in [0, 1]")
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if num_trees <= 0 or num_trees > num_perm:
            raise ValueError("num_trees must be in [1, num_perm]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if index_batch_size <= 0:
            raise ValueError("index_batch_size must be positive")

        self.threshold = float(sim_threshold)
        self.num_perm = int(num_perm)
        self.shingle_size = int(shingle_size)
        self.num_trees = int(num_trees)
        self.top_k = int(top_k)
        self.index_batch_size = int(index_batch_size)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes

        self.forest = MinHashLSHForest(num_perm=self.num_perm, l=self.num_trees)
        self._shingles: Dict[str, Set[str]] = {}
        self._pending_doc_ids: List[str] = []
        self._indexed_dirty = False
        self._pending_inserts = 0
        self.stats = Counter()
        self.jaccard_histogram = Counter()
        self.started_at = time.perf_counter()

    def _cache_path(self, doc_id: str) -> str:
        return os.path.join(self.minhash_root, f"{doc_id}.pkl")

    def build_minhash(self, shingles: Iterable[str]) -> MinHash:
        # datasketch.MinHash uses a stable SHA-1 based hash function by default.
        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8", errors="ignore"))
        return mh

    def _load_or_prepare(self, text: str, doc_id: str) -> Tuple[Set[str], MinHash]:
        shingles = make_shingles(text, self.shingle_size)
        if not shingles:
            raise ValueError(f"empty document after preprocessing: {doc_id}")

        mh_pkl = self._cache_path(doc_id)
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                minhash = pickle.load(f)
            if not isinstance(minhash, MinHash):
                raise TypeError(f"failed to parse MinHash cache: {mh_pkl}")
            return shingles, minhash

        minhash = self.build_minhash(shingles)
        with open(mh_pkl, "wb") as f:
            pickle.dump(minhash, f)
        return shingles, minhash

    def _ensure_searchable(self) -> None:
        if self._indexed_dirty:
            t0 = time.perf_counter()
            self.forest.index()
            self.stats["index_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
            self.stats["index_rebuilds"] += 1
            self._indexed_dirty = False
            self._pending_inserts = 0
            self._pending_doc_ids.clear()

    def _insert(self, doc_id: str, shingles: Set[str], minhash: MinHash) -> None:
        self.forest.add(doc_id, minhash)
        self._shingles[doc_id] = shingles
        self._pending_doc_ids.append(doc_id)
        self._indexed_dirty = True
        self._pending_inserts += 1
        self.stats["inserted_docs"] += 1
        if self._pending_inserts >= self.index_batch_size:
            self._ensure_searchable()

    def _candidate_ids(self, minhash: MinHash, current_doc_id: Optional[str] = None) -> List[str]:
        # LSH Forest is optimized for static batches: add() mutations are not
        # fully searchable until index() rebuilds sorted prefix tables. To keep
        # streaming dedup correct without rebuilding every insert, query the
        # last pending batch exactly as extra candidates.
        candidates = {str(candidate_id) for candidate_id in self.forest.query(minhash, self.top_k)}
        candidates.update(self._pending_doc_ids)
        if current_doc_id is not None:
            candidates.discard(current_doc_id)
        return list(candidates)

    def query(self, text: str, doc_id: str = "__query__") -> List[Tuple[str, float]]:
        shingles, minhash = self._load_or_prepare(text, doc_id)
        candidate_ids = self._candidate_ids(minhash, current_doc_id=doc_id)
        self.stats["candidate_pairs"] += len(candidate_ids)
        self.stats["queries"] += 1

        verified: List[Tuple[str, float]] = []
        for candidate_id in candidate_ids:
            score = jaccard(shingles, self._shingles[candidate_id])
            self.jaccard_histogram[round(score, 1)] += 1
            if score >= self.threshold:
                verified.append((candidate_id, score))
        return sorted(verified, key=lambda item: item[1], reverse=True)

    def deduplicate(self, text: str, id: int) -> bool:
        doc_id = str(id)
        t0 = time.perf_counter()
        shingles, minhash = self._load_or_prepare(text, doc_id)
        self.stats["signature_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        self.stats["processed_docs"] += 1

        t0 = time.perf_counter()
        candidate_ids = self._candidate_ids(minhash, current_doc_id=doc_id)
        self.stats["query_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        self.stats["candidate_pairs"] += len(candidate_ids)

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
            self._insert(doc_id, shingles, minhash)
            self.stats["insert_sec_x1e6"] += int((time.perf_counter() - t0) * 1_000_000)
        return is_duplicate

    def teardown(self) -> None:
        self._ensure_searchable()

    def summary(self) -> Dict[str, object]:
        elapsed = time.perf_counter() - self.started_at
        index_bucket_count = sum(len(table) for table in self.forest.hashtables)
        index_entry_count = sum(len(keys) for table in self.forest.hashtables for keys in table.values())
        signature_bytes = int(self.stats["inserted_docs"]) * self.num_perm * 8
        estimated_index_bytes = index_entry_count * (self.forest.k * 8 + 8) + index_bucket_count * 72
        return {
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "num_trees": self.num_trees,
            "prefix_depth": self.forest.k,
            "top_k": self.top_k,
            "index_batch_size": self.index_batch_size,
            "shingle_size": self.shingle_size,
            "processed_docs": int(self.stats["processed_docs"]),
            "inserted_docs": int(self.stats["inserted_docs"]),
            "predicted_duplicates": int(self.stats["predicted_duplicates"]),
            "candidate_pairs": int(self.stats["candidate_pairs"]),
            "index_rebuilds": int(self.stats["index_rebuilds"]),
            "index_bucket_count": int(index_bucket_count),
            "index_entry_count": int(index_entry_count),
            "signature_bytes": signature_bytes,
            "estimated_index_bytes": int(estimated_index_bytes),
            "peak_memory_mb": round(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 4),
            "runtime_sec": elapsed,
            "signature_sec": self.stats["signature_sec_x1e6"] / 1_000_000,
            "query_sec": self.stats["query_sec_x1e6"] / 1_000_000,
            "insert_sec": self.stats["insert_sec_x1e6"] / 1_000_000,
            "index_sec": self.stats["index_sec_x1e6"] / 1_000_000,
            "jaccard_histogram": {str(k): int(v) for k, v in sorted(self.jaccard_histogram.items())},
        }

    def write_stats(self, stats_json: str, stats_csv: str) -> None:
        summary = self.summary()
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            flat_summary = {k: v for k, v in summary.items() if not isinstance(v, dict)}
            writer = csv.DictWriter(f, fieldnames=list(flat_summary.keys()))
            writer.writeheader()
            writer.writerow(flat_summary)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash LSH Forest on a benchmark_dfs dataset.")
    parser.add_argument("--input", required=True, help="Benchmark tag, e.g. test_p_0.1")
    parser.add_argument("--sim-threshold", "--threshold", dest="threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--num-trees", "--l", dest="num_trees", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--shingle-size", "--ngram", dest="shingle_size", type=int, default=1)
    parser.add_argument("--index-batch-size", type=int, default=100)
    parser.add_argument("--force-compute-minhash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")

    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_forest_results")
    minhash_root = os.path.join(
        result_dir,
        "minhashes",
        f"perm{args.num_perm}_shingle{args.shingle_size}",
    )
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    run_tag = f"lsh_forest_{args.threshold}_{args.num_perm}_l{args.num_trees}_top{args.top_k}"
    output_file = os.path.join(result_dir, f"{run_tag}_preds.csv")
    result_file = os.path.join(result_dir, f"{run_tag}_score.csv")
    stats_json = os.path.join(result_dir, f"{run_tag}_stats.json")
    stats_csv = os.path.join(result_dir, f"{run_tag}_stats.csv")

    deduper = MinHashLSHForestDeduper(
        sim_threshold=args.threshold,
        num_perm=args.num_perm,
        shingle_size=args.shingle_size,
        minhash_root=minhash_root,
        num_trees=args.num_trees,
        top_k=args.top_k,
        index_batch_size=args.index_batch_size,
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
