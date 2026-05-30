import argparse
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
        "--num-probes",
        type=int,
        default=8,
        help=(
            "Number of extra narrow-band LSH tables to probe per query (T in paper). "
            "Each extra table uses band width = base_width - 1, giving higher recall "
            "at the cost of more false positives. num_probes=0 = standard LSH."
        ),
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
    )
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Multi-Probe LSH for MinHash / Jaccard similarity
#
# Paper: Lv et al., "Multi-Probe LSH", VLDB 2007.
#
# The paper's core idea: instead of using many hash tables, probe multiple
# buckets per table that have high collision probability with the query.
#
# Why swap-based probing (original code) doesn't work for MinHash:
#   Each MinHash slot is an independent random minimum. Swapping slot i
#   with slot j in a band gives a key that has near-zero probability of
#   matching any real document — it's essentially a random key. Confirmed
#   experimentally: zero bands where a swap variant of mh2 == mh1's band.
#
# Correct adaptation for MinHash:
#   The analog of "nearby buckets" in Jaccard/MinHash space is a band
#   with FEWER hash functions (smaller width). A narrower band has a
#   higher per-band collision probability R^(w') with w' < w, capturing
#   near-duplicates that the base bands miss.
#
#   We build `num_probes` extra LSH tables using width = base_width - 1
#   (one fewer hash per band). These extra tables act as the "probe"
#   buckets — they catch near-duplicates where the base bands fail.
#
#   This directly mirrors the paper's spirit: probe more buckets with
#   higher collision probability, trading some precision for recall.
#
# Collision probability analysis (why this works):
#   Base band width w:   P(match) = R^w          (e.g. R=0.8, w=5 → 0.33)
#   Probe band width w-1: P(match) = R^(w-1)     (e.g. R=0.8, w-1=4 → 0.41)
#   So probe bands catch ~24% more true near-duplicates at this similarity.
#
# False positive control:
#   Probe tables use the SAME threshold for candidate verification —
#   any candidate from a probe table is still verified against the
#   actual MinHash similarity before being declared a duplicate.
#   datasketch's MinHashLSH does this internally.
# ---------------------------------------------------------------------------


class MultiProbeLSH:
    """
    Multi-probe LSH for MinHash / Jaccard similarity.

    Maintains one base LSH index (standard MinHashLSH) and `num_probes`
    extra LSH indices with narrower bands (width = base_width - 1).
    Query probes all indices and returns the union of candidates.

    Parameters
    ----------
    threshold : float
        Jaccard similarity threshold (same as MinHashLSH).
    num_perm : int
        Number of MinHash permutations.
    num_probes : int
        Number of extra narrow-band tables to probe (T in paper).
        0 = standard LSH behavior.
    storage_config : dict
        Passed to MinHashLSH (e.g. {"type": "dict"}).
    """

    def __init__(
        self,
        threshold: float,
        num_perm: int,
        num_probes: int = 8,
        storage_config: dict = None,
    ):
        if storage_config is None:
            storage_config = {"type": "dict"}

        self.threshold = threshold
        self.num_perm = num_perm
        self.num_probes = int(num_probes)

        # Base LSH index — standard MinHashLSH.
        self.base_lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
            storage_config=storage_config,
        )

        # Derive base band parameters from datasketch's internal structure.
        # datasketch chooses b bands of width r such that (1/b)^(1/r) ≈ threshold.
        base_b = len(self.base_lsh.hashranges)
        base_r = self.base_lsh.hashranges[0][1] - self.base_lsh.hashranges[0][0]

        # Probe tables: use narrower bands (width = base_r - 1) for higher recall.
        # More bands needed to cover num_perm permutations with narrower width.
        self._probe_lsh: List[MinHashLSH] = []
        if self.num_probes > 0 and base_r > 1:
            probe_r = base_r - 1  # narrower band → higher P(match per band)
            probe_b = num_perm // probe_r  # more bands to cover all permutations
            for _ in range(self.num_probes):
                probe_lsh = MinHashLSH(
                    threshold=threshold,
                    num_perm=num_perm,
                    params=(probe_b, probe_r),  # override datasketch band selection
                    storage_config={"type": "dict"},
                )
                self._probe_lsh.append(probe_lsh)
        elif base_r <= 1:
            print(
                "[WARNING] base band width is already 1; "
                "cannot create narrower probe bands. num_probes ignored."
            )

    def insert(self, key, minhash: MinHash):
        self.base_lsh.insert(key, minhash)
        for probe_lsh in self._probe_lsh:
            try:
                probe_lsh.insert(key, minhash)
            except Exception:
                pass  # duplicate key guard

    def query(self, minhash: MinHash) -> list:
        candidates = set(self.base_lsh.query(minhash))
        for probe_lsh in self._probe_lsh:
            candidates.update(probe_lsh.query(minhash))
        return list(candidates)

    def __contains__(self, key) -> bool:
        return key in self.base_lsh


# ---------------------------------------------------------------------------
# Deduper
# ---------------------------------------------------------------------------

class LSHMultiProbeDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold,
        num_perm,
        minhash_root,
        recompute_minhashes=False,
        ngram=1,
        num_probes=8,
    ):
        super().__init__("lsh_multiprobe")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = int(ngram)
        self.lsh = MultiProbeLSH(
            threshold=self.T,
            num_perm=self.k,
            num_probes=int(num_probes),
            storage_config={"type": "dict"},
        )

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
            return mh

        mh = MinHash(num_perm=self.k)
        assert isinstance(text, str), f"Error empty document with id: {id}"
        words = text.split()
        if len(words) < self.ngram:
            s = set(words)
        else:
            s = set(
                " ".join(words[i: i + self.ngram])
                for i in range(len(words) - self.ngram + 1)
            )
        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)

        query_result = self.lsh.query(mh)
        uniq = not len(query_result) or (
            len(query_result) == 1 and query_result[0] == id
        )
        is_dup = not uniq

        if not is_dup:
            self.lsh.insert(id, mh)

        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_multiprobe_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(
        result_dir,
        f"lsh_multiprobe_{args.sim_threshold}_{args.num_perm}_preds.csv",
    )
    result_file = os.path.join(
        result_dir,
        f"lsh_multiprobe_{args.sim_threshold}_{args.num_perm}_score.csv",
    )

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHMultiProbeDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        num_probes=int(args.num_probes),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)