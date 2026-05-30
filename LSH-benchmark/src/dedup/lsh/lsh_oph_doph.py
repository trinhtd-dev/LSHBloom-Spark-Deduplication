import argparse
import hashlib
import os
import pickle
import sys
from typing import List

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../../synthetic_benchmark")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument(
        "--oph-bins",
        type=int,
        default=16,
        help="Number of bins used by the one-permutation sketch (must be <= num-perm).",
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
        help="Force recomputing minhashes instead of using cached values.",
    )
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2-universal hash helper (Carter & Wegman, 1977)
# Used by Optimal Densification (Shrivastava, ICML 2017, Section 3.1)
# huniv(x) = ((a*x + b) mod p) mod k
# ---------------------------------------------------------------------------

# A large Mersenne prime greater than any bin index we'll ever use.
_MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1


def _make_2universal(k: int, seed: int) -> "callable":
    """
    Returns a 2-universal hash function huniv: [l] -> [k].
    Parameters a, b are drawn deterministically from seed so that
    the same densification is reproducible across calls.
    """
    rng = np.random.default_rng(seed)
    a = int(rng.integers(1, _MERSENNE_PRIME))
    b = int(rng.integers(0, _MERSENNE_PRIME))
    p = _MERSENNE_PRIME

    def huniv(x: int) -> int:
        return int(((a * x + b) % p) % k)

    return huniv


# ---------------------------------------------------------------------------
# OPH Sketch with Optimal Densification
# Paper: Shrivastava, "Optimal Densification for Fast and Accurate Minwise
#         Hashing", ICML 2017. https://arxiv.org/abs/1703.04664
#
# Key ideas from the paper:
#   1. Hash tokens into k bins, take the minimum in each bin (OPH step).
#   2. For empty bins: instead of circular rotation (biased), use a
#      2-universal hash function to map each empty bin i to a random
#      non-empty bin j = huniv(i), then take the value from bin j.
#      This is the "Optimal Densification" scheme (Algorithm 1 in paper).
#   3. The resulting k values are directly usable as independent-ish
#      minwise hash values with the LSH collision probability = Jaccard(S1,S2).
# ---------------------------------------------------------------------------

class OPHOptimalDenseSketch:
    """
    One-Permutation Hashing with Optimal Densification.

    Follows Algorithm 1 of Shrivastava (ICML 2017):
      - num_bins (k) bins, each holds min hash of tokens that fall in it.
      - Empty bins are filled via a 2-universal hash that maps them to a
        random non-empty bin, preserving the LSH collision probability.
      - The output has exactly num_bins values (one per bin).

    NOTE: num_perm must equal num_bins for this to be theoretically correct.
    Tiling (expanding num_bins -> num_perm by repetition) introduces
    correlations that break the LSH variance guarantees; see the paper.
    """

    def __init__(self, num_bins: int, seed: int = 42):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self.num_bins = int(num_bins)
        self._seed = seed
        # Pre-build the 2-universal hash for densification.
        # huniv maps an empty bin index -> a bin index in [0, num_bins).
        self._huniv = _make_2universal(self.num_bins, seed=seed)

    def _token_hash(self, token: str) -> int:
        """Hash a token to a 64-bit unsigned integer."""
        digest = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"oph-opt")
        return int.from_bytes(digest.digest(), "big", signed=False)

    def _bin_of(self, token_hash: int) -> int:
        """Assign token to a bin by uniform range partition."""
        # Equivalent to floor(h * k / 2^64) but using modulo for simplicity.
        # For uniform hashes this gives approximately uniform bin assignment.
        return (token_hash * self.num_bins) >> 64
    def compute(self, text: str, ngram: int = 1) -> np.ndarray:
        """
        Compute the densified OPH sketch for a text.

        Returns an array of shape (num_bins,) with dtype uint64.
        Each value is the minimum hash in that bin (after densification).
        """
        words = text.split()
        if len(words) < ngram:
            tokens = set(words)
        else:
            tokens = set(
                " ".join(words[i: i + ngram]) for i in range(len(words) - ngram + 1)
            )
        if not tokens:
            raise ValueError("Cannot build sketch for empty text.")

        # --- Step 1: OPH — place min-hash into each bin ---
        INF = np.uint64(2**64 - 1)
        bin_min = np.full(self.num_bins, INF, dtype=np.uint64)

        for tok in tokens:
            h = np.uint64(self._token_hash(tok))
            b = self._bin_of(int(h))
            if h < bin_min[b]:
                bin_min[b] = h

        occupied = bin_min < INF  # bool mask of non-empty bins

        if not occupied.any():
            raise ValueError("All bins empty — cannot densify.")

        # --- Step 2: Optimal Densification (Shrivastava ICML 2017) ---
        # For each empty bin i, set bin_min[i] = bin_min[huniv(i)],
        # where huniv is a 2-universal hash mapping i -> a bin index.
        # If huniv(i) is also empty, keep re-sampling until we hit a
        # non-empty bin (cycle-free because we only sample once and
        # fall back to the closest non-empty if still empty).
        #
        # Simple single-sample version (O(k) total, matches paper):
        dense = bin_min.copy()
        non_empty_indices = np.where(occupied)[0]

        for i in range(self.num_bins):
            if occupied[i]:
                continue
            # Map i to a candidate bin via 2-universal hash.
            j = self._huniv(i)
            if occupied[j]:
                dense[i] = bin_min[j]
            else:
                # Fallback: find the nearest non-empty bin (circular right),
                # which ensures no bin is left as INF.
                for step in range(1, self.num_bins):
                    jj = (j + step) % self.num_bins
                    if occupied[jj]:
                        dense[i] = bin_min[jj]
                        break

        return dense  # shape (num_bins,), dtype uint64


# ---------------------------------------------------------------------------
# Bridge: convert our sketch values into a datasketch MinHash object.
#
# The key fix vs. the original code:
#   datasketch MinHash uses its own internal hash permutation scheme.
#   We CANNOT simply assign our values to mh.hashvalues because the
#   internal format is different (datasketch stores uint32 minimums from
#   (a*x+b mod p) mod 2^32 for each permutation).
#
#   The correct approach: treat num_bins == num_perm, bypass datasketch's
#   hash computation entirely by subclassing/monkey-patching hashvalues
#   with properly scaled values.  datasketch's MinHashLSH only compares
#   hashvalues arrays directly, so as long as BOTH query and stored
#   MinHash objects were built the same way, the collision probability
#   is preserved.
#
#   We scale our uint64 values down to uint32 range (right-shift by 32)
#   so they fit datasketch's expected dtype without overflow.
# ---------------------------------------------------------------------------

def sketch_to_minhash(signature: np.ndarray, num_perm: int) -> MinHash:
    """
    Convert a densified OPH signature (uint64 array of length num_perm)
    into a datasketch MinHash object suitable for MinHashLSH.

    We scale to uint32 to match datasketch's internal representation.
    The relative ordering of values is preserved, so collision probability
    (= Jaccard similarity) is maintained.
    """
    if len(signature) != num_perm:
        raise ValueError(
            f"Signature length {len(signature)} != num_perm {num_perm}"
        )
    mh = MinHash(num_perm=num_perm)
    # Right-shift by 32 to map uint64 -> uint32 range uniformly.
    mh.hashvalues = (signature >> np.uint64(32)).astype(np.uint32)
    return mh


# ---------------------------------------------------------------------------
# Deduper
# ---------------------------------------------------------------------------

class LSHOphDophDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        minhash_root: str,
        recompute_minhashes: bool = False,
        ngram: int = 1,
        oph_bins: int = 16,
    ):
        super().__init__("lsh_oph_doph")
        self.T = float(sim_threshold)
        # Per the paper, num_perm should equal num_bins for correct LSH.
        # If they differ we warn and clamp.
        if num_perm != oph_bins:
            print(
                f"[WARNING] num_perm ({num_perm}) != oph_bins ({oph_bins}). "
                "Tiling bins to fill num_perm breaks LSH variance guarantees. "
                "Setting num_perm = oph_bins for correctness."
            )
            num_perm = oph_bins

        self.k = int(num_perm)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = int(ngram)
        self.sketch = OPHOptimalDenseSketch(num_bins=self.k)
        self.lsh = MinHashLSH(
            threshold=self.T, num_perm=self.k, storage_config={"type": "dict"}
        )
        self._inserted_ids: set = set()

    def deduplicate(self, text: str, id: int) -> bool:
        signature = self.sketch.compute(text, self.ngram)
        mh = sketch_to_minhash(signature, self.k)

        is_dup = bool(self.lsh.query(mh))
        if not is_dup:
            if id not in self._inserted_ids:
                self.lsh.insert(id, mh)
                self._inserted_ids.add(id)
        return is_dup


if __name__ == "__main__":
    args = get_args()

    # num_perm is forced equal to oph_bins inside the deduper for correctness.
    effective_num_perm = args.oph_bins

    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_oph_doph_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{effective_num_perm}")
    output_file = os.path.join(
        result_dir,
        f"lsh_oph_doph_{args.sim_threshold}_{effective_num_perm}_preds.csv",
    )
    result_file = os.path.join(
        result_dir,
        f"lsh_oph_doph_{args.sim_threshold}_{effective_num_perm}_score.csv",
    )

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHOphDophDeduper(
        sim_threshold=args.sim_threshold,
        num_perm=args.num_perm,
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=args.ngram,
        oph_bins=args.oph_bins,
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
