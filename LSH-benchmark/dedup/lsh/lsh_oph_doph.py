import argparse
import hashlib
import math
import os
import pickle
import sys
from typing import List, Sequence

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
        "--oph-groups",
        type=int,
        default=16,
        help="Number of OPH groups used to create a sparse one-permutation style sketch.",
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
        help="Force recomputing minhashes instead of using cached values.",
    )
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


class OPHSketch:
    """OPH/DOPH-style sketch that outputs exactly num_perm values."""

    def __init__(self, num_perm: int, groups: int = 16):
        if groups < 2:
            raise ValueError("groups must be >= 2")
        if num_perm < groups:
            raise ValueError("num_perm must be >= groups")
        self.num_perm = num_perm
        self.groups = groups
        self.per_group = int(math.ceil(num_perm / groups))

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"oph-doph")
        return int.from_bytes(digest.digest(), "big", signed=False)

    def compute(self, text: str, ngram: int) -> List[int]:
        words = text.split()
        if len(words) < ngram:
            tokens = set(words)
        else:
            tokens = set(" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1))

        if not tokens:
            raise ValueError("Cannot build sketch for empty text")

        buckets: List[List[int]] = [[] for _ in range(self.groups)]
        for tok in tokens:
            h = self._hash_token(tok)
            buckets[h % self.groups].append(h)

        signature = []
        for b in buckets:
            if b:
                vals = sorted(b)[: self.per_group]
                if len(vals) < self.per_group:
                    vals.extend([None] * (self.per_group - len(vals)))
            else:
                vals = [None] * self.per_group
            signature.extend(vals)

        signature = signature[: self.num_perm]
        if len(signature) < self.num_perm:
            signature.extend([None] * (self.num_perm - len(signature)))
        return self._densify(signature)

    def _densify(self, signature: Sequence[int]) -> List[int]:
        dense = list(signature)
        for i, value in enumerate(dense):
            if value is not None:
                continue
            for step in range(1, len(dense) + 1):
                j = (i + step) % len(dense)
                if dense[j] is not None:
                    dense[i] = self._fill_value(i, dense[j], step)
                    break
            if dense[i] is None:
                dense[i] = self._fill_value(i, 0, len(dense))
        return [int(x) for x in dense]

    def _fill_value(self, idx: int, value: int, step: int) -> int:
        digest = hashlib.blake2b(digest_size=8, person=b"doph-fill")
        digest.update(int(idx).to_bytes(4, "little", signed=False))
        digest.update(int(step).to_bytes(4, "little", signed=False))
        digest.update(int(value).to_bytes(8, "little", signed=False))
        return int.from_bytes(digest.digest(), "big", signed=False)


class LSHOphDophDeduper(DedupHarness):
    def __init__(self, sim_threshold, num_perm, minhash_root, recompute_minhashes=False, ngram=1, oph_groups=16):
        super().__init__("lsh_oph_doph")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = int(ngram)
        self.oph = OPHSketch(num_perm=self.k, groups=int(oph_groups))
        self.lsh = MinHashLSH(threshold=self.T, num_perm=self.k, storage_config={"type": "dict"})

    def oph_to_minhash(self, signature):
        mh = MinHash(num_perm=self.k)
        mh.hashvalues = np.array(signature, dtype=np.uint64)
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
        oph_sig = self.oph.compute(text, self.ngram)
        mh = self.oph_to_minhash(oph_sig)

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
        oph_groups=int(args.oph_groups),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
