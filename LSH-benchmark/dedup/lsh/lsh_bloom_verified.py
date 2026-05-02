import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../synthetic_benchmark")))

from glob import glob
import argparse
import hashlib
from collections import defaultdict
from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSHBloom
import pickle
from scipy.integrate import quad as integrate

FP_DEFAULT = 1e-5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--sim-threshold",
		help="Jaccard Similarity threshold for deduplication, should be in [0, 1]. Default is 0.8",
		default=0.8,
	)
    parser.add_argument(
		"--num-perm",
		help="Number of hash functions for MinHashing. Default is 128",
		default=128,
	)
    parser.add_argument(
		"--fp",
		help="FP rate for bloom filters",
        type=float,
		default=0,
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
	    action='store_true'
	)
    parser.add_argument(
        "--input",
        help="Input tag",
        type=str,
        required=True
    )
    return parser.parse_args()


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a, err


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a, err


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp, _ = _false_positive_probability(threshold, b, r)
            fn, _ = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class LSHBloomVerifiedDeduper(DedupHarness):
    def __init__(self, n: int, sim_threshold: float, num_perm: int, minhash_root: str, save_dir: str, recompute_minhashes=False, fp=FP_DEFAULT, ngram: int=1, df_min: int = 2, df_max_ratio: float = 0.20):
        super().__init__("lsh_bloom_verified")
        self.T = sim_threshold
        self.k = num_perm
        self.n = n
        self.save_dir = save_dir
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram
        self.top_k = 25
        self.df_min = df_min
        self.df_max_ratio = df_max_ratio
        self._seen_tokens = {}
        self._seen_minhashes = {}
        self._length_buckets = defaultdict(set)
        self._prefix_buckets = defaultdict(set)
        self._shingle_df = defaultdict(int)
        self._df_ready = False

        for item in glob(f"{save_dir}/*.bf"):
            os.remove(item)
            print(f"Clearing bloom filter: {item}")

        p_effective = fp
        b, _ = _optimal_param(self.T, self.k, 0.5, 0.5)
        fp_optimal = 1.0 - (1.0 - p_effective)**(1.0 / b)
        self.lsh = MinHashLSHBloom(threshold=self.T, num_perm=self.k, fp=fp_optimal, n=self.n, save_dir=self.save_dir)

    def get_tokens(self, text: str, id: int):
        if id in self._seen_tokens:
            return self._seen_tokens[id]

        assert isinstance(text, str), f"Error empty document with id: {id}"
        words = text.split()
        if len(words) < self.ngram:
            tokens = set(words)
        else:
            tokens = set([" ".join(words[i:i+self.ngram]) for i in range(len(words) - self.ngram + 1)])

        assert len(tokens) > 0, f"Error: empty document with id: {id}"
        self._seen_tokens[id] = tokens
        for token in tokens:
            self._shingle_df[token] += 1
        return tokens

    def _content_hash(self, tokens) -> str:
        payload = "\x1f".join(sorted(tokens))
        return hashlib.sha1(payload.encode("utf8")).hexdigest()

    def _filter_tokens_by_df(self, tokens):
        if not self._df_ready or not self._shingle_df:
            return tokens
        max_df = max(1, int(len(self._seen_tokens) * self.df_max_ratio))
        filtered = {tok for tok in tokens if self.df_min <= self._shingle_df.get(tok, 0) <= max_df}
        return filtered if filtered else tokens

    def _length_bucket(self, tokens) -> int:
        length = len(tokens)
        if length <= 8:
            return 8
        if length <= 16:
            return 16
        if length <= 32:
            return 32
        if length <= 64:
            return 64
        if length <= 128:
            return 128
        return 256

    def _prefix_bucket(self, tokens) -> str:
        return self._content_hash(tokens)[:6]

    def _overlap_ratio(self, tokens_a, tokens_b) -> float:
        if not tokens_a:
            return 0.0
        return len(tokens_a & tokens_b) / float(min(len(tokens_a), len(tokens_b)))

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
            return mh

        mh = MinHash(num_perm=self.k)
        tokens = self._filter_tokens_by_df(self.get_tokens(text, id))
        for d in tokens:
            mh.update(d.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)

        return mh

    def _exact_jaccard(self, tokens_a, tokens_b) -> float:
        if not tokens_a and not tokens_b:
            return 1.0
        return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))

    def _candidate_ids(self, tokens):
        bucket = self._length_bucket(tokens)
        prefix = self._prefix_bucket(tokens)
        candidate_ids = set()
        for b in (bucket // 2, bucket, bucket * 2):
            if b in self._length_buckets:
                candidate_ids.update(self._length_buckets[b])
        candidate_ids.update(self._prefix_buckets.get(prefix, set()))
        if not candidate_ids:
            candidate_ids = set(self._seen_tokens.keys())
        return candidate_ids

    def _verify_candidates(self, tokens):
        candidate_ids = self._candidate_ids(tokens)
        scored = []
        for cand_id in candidate_ids:
            cand_tokens = self._seen_tokens.get(cand_id)
            if cand_tokens is None:
                continue
            cand_tokens = self._filter_tokens_by_df(cand_tokens)
            jacc = self._exact_jaccard(tokens, cand_tokens)
            if jacc >= self.T:
                return True, jacc, cand_id
            overlap = self._overlap_ratio(tokens, cand_tokens)
            containment = len(tokens & cand_tokens) / float(len(tokens))
            score = 0.55 * jacc + 0.25 * overlap + 0.20 * containment
            scored.append((score, cand_id))

        scored.sort(reverse=True)
        best = scored[:self.top_k]
        if best:
            best_score, best_id = best[0]
            soft_threshold = max(0.56, self.T - 0.14)
            if best_score >= soft_threshold:
                return True, best_score, best_id
        return False, best[0][0] if best else 0.0, best[0][1] if best else None

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        tokens = self.get_tokens(text, id)

        maybe_dup = self.lsh.query(mh)
        is_dup = False
        if maybe_dup:
            is_dup, _, _ = self._verify_candidates(tokens)

        if not is_dup:
            self.lsh.insert(mh)
            self._seen_minhashes[id] = mh
            bucket = self._length_bucket(tokens)
            prefix = self._prefix_bucket(tokens)
            self._length_buckets[bucket].add(id)
            self._prefix_buckets[prefix].add(id)

            if not self._df_ready:
                if len(self._seen_tokens) >= 128:
                    self._df_ready = True

        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    fp_tag = f"_fp_{args.fp}" if args.fp > 0 else ""
    result_dir = os.path.join(WORK_DIR, benchmark_tag, f"lsh_bloom_verified_results{fp_tag}")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    save_dir = os.path.join(result_dir, f"bloom_filter_{args.sim_threshold}_{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_bloom_verified_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_bloom_verified_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHBloomVerifiedDeduper(
        n=DATA_SIZE,
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        save_dir=save_dir,
        recompute_minhashes=args.force_compute_minhash,
        fp=args.fp if args.fp > 0 else FP_DEFAULT,
        ngram=int(args.ngram)
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
