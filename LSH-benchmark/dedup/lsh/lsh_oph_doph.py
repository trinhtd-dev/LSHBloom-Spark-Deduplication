import argparse
import hashlib
import os
import pickle
import sys
from typing import List

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../synthetic_benchmark")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-threshold", default=0.8)
    parser.add_argument("--num-perm", default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument(
        "--oph-bins",
        type=int,
        default=16,
        help="Number of bins used by the one-permutation sketch.",
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
        help="Force recomputing minhashes instead of using cached values.",
    )
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


class OPHDOPHSketch:
    """Paper-style OPH/DOPH sketch: one permutation, bins, densification."""

    def __init__(self, num_perm: int, num_bins: int):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if num_perm < num_bins:
            raise ValueError("num_perm must be >= num_bins")
        self.num_perm = int(num_perm)
        self.num_bins = int(num_bins)
        self.bins_per_bucket = int(np.ceil(self.num_perm / self.num_bins))

    def _token_hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"oph-doph")
        return int.from_bytes(digest.digest(), "big", signed=False)

    def _one_perm_bin(self, token_hash: int) -> int:
        return token_hash % self.num_bins

    def compute(self, text: str, ngram: int) -> np.ndarray:
        words = text.split()
        if len(words) < ngram:
            tokens = set(words)
        else:
            tokens = set(" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1))

        if not tokens:
            raise ValueError("Cannot build sketch for empty text")

        bins: List[List[int]] = [[] for _ in range(self.num_bins)]
        for tok in tokens:
            h = self._token_hash(tok)
            bins[self._one_perm_bin(h)].append(h)

        signature = np.full(self.num_bins, np.uint64(2**64 - 1), dtype=np.uint64)
        occupied = np.zeros(self.num_bins, dtype=bool)
        for i, bucket in enumerate(bins):
            if bucket:
                signature[i] = np.uint64(min(bucket))
                occupied[i] = True

        dense_bins = self._densify(signature, occupied)
        return self._expand_to_num_perm(dense_bins)

    def _densify(self, signature: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        dense = signature.copy()
        for i in range(self.num_bins):
            if occupied[i]:
                continue
            for step in range(1, self.num_bins + 1):
                j = (i + step) % self.num_bins
                if occupied[j]:
                    dense[i] = self._densify_value(i, dense[j], step)
                    break
            if dense[i] == np.uint64(2**64 - 1):
                dense[i] = self._densify_value(i, np.uint64(0), self.num_bins)
        return dense

    def _expand_to_num_perm(self, dense_bins: np.ndarray) -> np.ndarray:
        if self.num_perm == self.num_bins:
            return dense_bins
        repeats = int(np.ceil(self.num_perm / self.num_bins))
        expanded = np.tile(dense_bins, repeats)[: self.num_perm]
        return expanded.astype(np.uint64)

    def _densify_value(self, idx: int, value: np.uint64, step: int) -> np.uint64:
        digest = hashlib.blake2b(digest_size=8, person=b"oph-doph-dense")
        digest.update(int(idx).to_bytes(4, "little", signed=False))
        digest.update(int(step).to_bytes(4, "little", signed=False))
        digest.update(int(value).to_bytes(8, "little", signed=False))
        return np.uint64(int.from_bytes(digest.digest(), "big", signed=False))


class LSHOphDophDeduper(DedupHarness):
    def __init__(self, sim_threshold, num_perm, minhash_root, recompute_minhashes=False, ngram=1, oph_bins=16):
        super().__init__("lsh_oph_doph")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = int(ngram)
        self.sketch = OPHDOPHSketch(num_perm=self.k, num_bins=int(oph_bins))
        self.lsh = MinHashLSH(threshold=self.T, num_perm=self.k, storage_config={"type": "dict"})

    def sketch_to_minhash(self, signature: np.ndarray) -> MinHash:
        mh = MinHash(num_perm=self.k)
        if len(signature) != self.k:
            raise ValueError(f"Sketch length mismatch: expected {self.k}, got {len(signature)}")
        mh.hashvalues = np.asarray(signature, dtype=np.uint64)
        return mh

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            if isinstance(mh, MinHash):
                return mh

        mh = MinHash(num_perm=self.k)
        assert isinstance(text, str), f"Error empty document with id: {id}"
        words = text.split()
        if len(words) < self.ngram:
            s = set(words)
        else:
            s = set([" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)])
        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        sketch = self.sketch.compute(text, self.ngram)
        mh = self.sketch_to_minhash(sketch)
        is_dup = bool(self.lsh.query(mh))
        if not is_dup:
            self.lsh.insert(id, mh)
        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_oph_doph_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_oph_doph_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_oph_doph_{args.sim_threshold}_{args.num_perm}_score.csv")

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHOphDophDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        oph_bins=int(args.oph_bins),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
