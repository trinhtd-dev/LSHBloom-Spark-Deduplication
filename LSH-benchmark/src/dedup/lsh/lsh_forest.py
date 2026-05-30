import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../../synthetic_benchmark")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

import argparse
import pickle
from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSHForest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-threshold",
        help="Jaccard Similarity threshold for deduplication, should be in [0, 1]. Default is 0.8",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--num-perm",
        help="Number of hash functions for MinHashing. Default is 128",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-trees",
        help="Number of prefix trees in the LSH Forest (l in the paper). Default is 8.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ngram",
        help="N-gram size for MinHashing. Default is 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--force-compute-minhash",
        help="Whether to force computing minhashes as opposed to reading cached minhashes from disk",
        action="store_true",
    )
    parser.add_argument(
        "--input",
        help="Input tag",
        type=str,
        required=True,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LSH Forest for MinHash / Jaccard similarity
#
# Paper: Bawa, Condie & Ganesan, "LSH Forest: Self-Tuning Indexes for
#        Similarity Search", WWW 2005.
#
# How LSH Forest differs from MinHashLSH (standard banding):
#
#   MinHashLSH (standard):
#     - Fixes band width r and number of bands b = num_perm / r upfront.
#     - A pair is a candidate iff they share at least one full band.
#     - The threshold is determined by (b, r): threshold ≈ (1/b)^(1/r).
#     - No adaptation to the actual data distribution.
#
#   LSH Forest:
#     - Builds l prefix trees over the MinHash values.
#     - Each tree stores documents indexed by their full hash label
#       (all num_perm/l hash values), structured as a prefix trie.
#     - At query time, DESCEND finds the deepest prefix level where
#       the query matches any stored document, then ascends level by
#       level collecting M candidates total across all trees.
#     - The effective band width adapts dynamically to the data:
#       dense regions → deeper match (shorter effective band) →
#       more candidates; sparse regions → shallower match.
#     - This "self-tuning" avoids fixing (b, r) in advance and works
#       better when data density varies across the similarity range.
#
# datasketch implementation note (from docs):
#   MinHashLSHForest fixes num_perm hash functions and sets the
#   maximum depth of each prefix tree to k = num_perm / l.
#   After all inserts, forest.index() must be called to build the
#   sorted arrays needed for prefix queries.
#
# Query API difference vs MinHashLSH:
#   MinHashLSH.query(mh)       → returns all candidates above threshold
#   MinHashLSHForest.query(mh, k) → returns top-k approximate neighbors
#
#   For deduplication we set k = num_candidates (configurable) and
#   declare a document duplicate if any returned neighbor has
#   estimated Jaccard >= threshold (re-verified with full MinHash).
# ---------------------------------------------------------------------------


class LSHForestDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        minhash_root: str,
        recompute_minhashes: bool = False,
        ngram: int = 1,
        num_trees: int = 8,
    ):
        super().__init__("lsh_forest")
        self.T = sim_threshold
        self.k = num_perm
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram

        # l = num_trees prefix trees; max depth per tree = num_perm / l
        self.forest = MinHashLSHForest(num_perm=self.k, l=num_trees)

        # Number of candidates to retrieve per query.
        # Paper retrieves M candidates total across all trees.
        # We set M = num_trees * (num_perm // num_trees) as a sensible default
        # — roughly proportional to dataset density like the paper suggests.
        self._num_candidates = num_trees * (num_perm // num_trees)

        # Store MinHash objects for similarity re-verification of candidates.
        # Unlike b-bit variant, here we need full MinHash for jaccard().
        self._minhash_store: dict[int, MinHash] = {}

        # Track whether forest needs re-indexing after new inserts.
        self._dirty = False

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
            return mh

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

        mh = MinHash(num_perm=self.k)
        for d in s:
            mh.update(d.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)

        # LSH Forest requires index() to be called after inserts before querying.
        # We rebuild the index lazily when the forest is dirty.
        if self._dirty:
            self.forest.index()
            self._dirty = False

        # Query: retrieve top-k approximate neighbors.
        # Unlike MinHashLSH, LSHForest.query returns the k closest neighbors
        # by prefix length — not all candidates above a threshold.
        candidates = self.forest.query(mh, self._num_candidates)

        # Re-verify candidates: check actual Jaccard against threshold.
        # This mirrors the paper's two-phase approach:
        #   phase 1 (forest query) → candidate set
        #   phase 2 (exact check) → confirmed duplicates
        is_dup = False
        for cand_id in candidates:
            if cand_id == id:
                continue
            cand_mh = self._minhash_store.get(cand_id)
            if cand_mh is None:
                continue
            if mh.jaccard(cand_mh) >= self.T:
                is_dup = True
                break

        if not is_dup:
            self.forest.add(id, mh)
            self._minhash_store[id] = mh
            self._dirty = True  # forest needs re-indexing before next query

        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_forest_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(
        result_dir,
        f"lsh_forest_{args.sim_threshold}_{args.num_perm}_l{args.num_trees}_preds.csv",
    )
    result_file = os.path.join(
        result_dir,
        f"lsh_forest_{args.sim_threshold}_{args.num_perm}_l{args.num_trees}_score.csv",
    )

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHForestDeduper(
        sim_threshold=args.sim_threshold,
        num_perm=args.num_perm,
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=args.ngram,
        num_trees=args.num_trees,
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
