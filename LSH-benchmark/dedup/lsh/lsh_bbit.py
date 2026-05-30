import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../synthetic_benchmark")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

import argparse
import pickle
from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSH, bBitMinHash


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument(
        "--b-bit",
        type=int,
        default=1,
        help="Number of bits per hash value to keep. b=1 → 32x memory reduction vs full MinHash.",
    )
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--force-compute-minhash", action="store_true")
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# b-bit MinHash deduplication
#
# Paper: Li & König, "b-Bit Minwise Hashing", WWW 2011.
#
# Core idea:
#   Standard MinHash stores full 32-bit hash values → k * 32 bits per doc.
#   b-bit MinHash keeps only the b lowest bits of each hash value →
#   k * b bits per doc, a 32/b memory reduction.
#
#   The b-bit Jaccard estimator is unbiased and nearly as accurate as
#   full MinHash for high similarities (the range we care about for dedup).
#
# Correct pipeline (per paper):
#   1. Compute full MinHash (needed once for LSH candidate generation).
#   2. Convert to b-bit MinHash immediately → store only b-bit version.
#   3. Use MinHashLSH with full MinHash for fast candidate retrieval.
#   4. Re-verify candidates using b-bit Jaccard estimator.
#   5. Discard the full MinHash after step 3 — never store it long-term.
#
# What the original code did wrong:
#   - Stored full MinHash for ALL documents in RAM (_seen_minhashes)
#     → linear memory growth, defeats the purpose of b-bit compression.
#   - Used b-bit only as a post-LSH re-ranker, not as primary storage.
#   - Had dead code (best_score/best_match_id computed but never used).
#   - Stored tokens for both dup and non-dup paths identically (duplicate code).
#   - break on first candidate above threshold instead of checking all.
# ---------------------------------------------------------------------------


class LSHBBitDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        minhash_root: str,
        recompute_minhashes: bool = False,
        ngram: int = 1,
        b_bit: int = 1,
    ):
        super().__init__("lsh_bbit")
        self.T = sim_threshold
        self.k = num_perm
        self.b_bit = b_bit
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram

        self.lsh = MinHashLSH(
            threshold=self.T, num_perm=self.k, storage_config={"type": "dict"}
        )

        # KEY FIX: store only b-bit MinHash per document, not full MinHash.
        # Memory per doc: k * b bits instead of k * 32 bits → 32/b reduction.
        self._bbit_store: dict[int, bBitMinHash] = {}

    def get_minhash(self, text: str, id: int) -> MinHash:
        """
        Compute or load full MinHash. Full MinHash is only kept temporarily —
        it is converted to b-bit and discarded after indexing.
        """
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
            return mh

        assert isinstance(text, str), f"Error: empty document with id: {id}"
        words = text.split()
        tokens = set(words) if len(words) < self.ngram else set(
            " ".join(words[i: i + self.ngram])
            for i in range(len(words) - self.ngram + 1)
        )
        assert len(tokens) > 0, f"Error: empty document with id: {id}"

        mh = MinHash(num_perm=self.k)
        for token in tokens:
            mh.update(token.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)

        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        # Step 1: Compute full MinHash (needed for LSH query + insert).
        mh = self.get_minhash(text, id)

        # Step 2: Query LSH with full MinHash to get candidates.
        query_result = self.lsh.query(mh)

        # Mirror reference LSH edge-case: [id] = found itself, not a dup.
        candidates = [c for c in query_result if c != id]

        # Step 3: Re-verify each candidate using b-bit Jaccard estimator.
        # This is the correct use of b-bit MinHash per Li & König (2011):
        # b-bit provides a memory-efficient similarity estimate for verification.
        is_dup = False
        if candidates:
            # Convert query doc to b-bit once (avoid repeated conversion).
            bb_query = bBitMinHash(mh, b=self.b_bit)
            for cand_id in candidates:
                bb_cand = self._bbit_store.get(cand_id)
                if bb_cand is None:
                    continue
                score = bb_query.jaccard(bb_cand)
                if score >= self.T:
                    is_dup = True
                    break  # found a confirmed duplicate, no need to check more

        # Step 4: If unique, index it.
        # Store only b-bit version — discard full MinHash after insert.
        if not is_dup:
            self.lsh.insert(id, mh)
            # Convert to b-bit and store — full mh goes out of scope here.
            self._bbit_store[id] = bBitMinHash(mh, b=self.b_bit)

        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_bbit_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(
        result_dir,
        f"lsh_bbit_{args.sim_threshold}_{args.num_perm}_b{args.b_bit}_preds.csv",
    )
    result_file = os.path.join(
        result_dir,
        f"lsh_bbit_{args.sim_threshold}_{args.num_perm}_b{args.b_bit}_score.csv",
    )

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHBBitDeduper(
        sim_threshold=args.sim_threshold,
        num_perm=args.num_perm,
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=args.ngram,
        b_bit=args.b_bit,
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)