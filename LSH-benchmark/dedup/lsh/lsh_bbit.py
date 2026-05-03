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
        "--b-bit",
        help="Number of bits to keep in b-bit MinHash. Default is 1",
        type=int,
        default=1,
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


class LSHBBitDeduper(DedupHarness):
    def __init__(self, sim_threshold: float, num_perm: int, minhash_root: str, recompute_minhashes=False, ngram: int = 1, b_bit: int = 1):
        super().__init__("lsh_bbit")
        self.T = sim_threshold
        self.k = num_perm
        self.b_bit = b_bit
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram
        self._seen_tokens = {}
        self._seen_minhashes = {}
        self._seen_bbit = {}
        self.lsh = MinHashLSH(threshold=self.T, num_perm=self.k, storage_config={"type": "dict"})

    def get_tokens(self, text: str, id: int):
        if id in self._seen_tokens:
            return self._seen_tokens[id]

        assert isinstance(text, str), f"Error empty document with id: {id}"
        words = text.split()
        if len(words) < self.ngram:
            tokens = set(words)
        else:
            tokens = set([" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)])

        assert len(tokens) > 0, f"Error: empty document with id: {id}"
        self._seen_tokens[id] = tokens
        return tokens

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
            return mh

        mh = MinHash(num_perm=self.k)
        for token in self.get_tokens(text, id):
            mh.update(token.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)

        return mh

    def get_bbit_minhash(self, mh: MinHash) -> bBitMinHash:
        return bBitMinHash(mh, b=self.b_bit)

    def _jaccard_with_bbit(self, mh_a: MinHash, mh_b: MinHash) -> float:
        bb_a = self.get_bbit_minhash(mh_a)
        bb_b = self.get_bbit_minhash(mh_b)
        return bb_a.jaccard(bb_b)

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        query_result = self.lsh.query(mh)

        is_dup = False
        best_match_id = None
        best_score = 0.0

        if query_result:
            for cand_id in query_result:
                cand_mh = self._seen_minhashes.get(cand_id)
                if cand_mh is None:
                    continue
                score = self._jaccard_with_bbit(mh, cand_mh)
                if score > best_score:
                    best_score = score
                    best_match_id = cand_id
                if score >= self.T:
                    is_dup = True
                    best_match_id = cand_id
                    break

        if not is_dup:
            self.lsh.insert(id, mh)
            self._seen_minhashes[id] = mh
            self._seen_tokens[id] = self.get_tokens(text, id)
        else:
            self._seen_tokens[id] = self.get_tokens(text, id)

        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_bbit_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_bbit_{args.sim_threshold}_{args.num_perm}_b{args.b_bit}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_bbit_{args.sim_threshold}_{args.num_perm}_b{args.b_bit}_score.csv")

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHBBitDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        b_bit=int(args.b_bit),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
