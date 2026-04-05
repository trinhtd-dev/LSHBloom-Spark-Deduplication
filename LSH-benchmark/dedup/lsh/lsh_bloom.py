import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from glob import glob
import argparse
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
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.

    Returns a tuple: (# bands, band size)
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp, pe = _false_positive_probability(threshold, b, r)
            fn, ne = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt

class LSHBloomDeduper(DedupHarness):
    def __init__(self, n: int, sim_threshold: float, num_perm: int, minhash_root: str, save_dir: str, recompute_minhashes=False, fp=FP_DEFAULT, ngram: int=1):
        super().__init__("lsh_bloom")
        self.T = sim_threshold
        self.k = num_perm
        self.n = n
        self.save_dir = save_dir
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram
        
        # clear save_dir if re-running
        for item in glob(f"{save_dir}/*.bf"):
            os.remove(item)
            print(f"Clearing bloom filter: {item}")

        # set true effective fp rate 
        p_effective = fp
        b, r = _optimal_param(self.T, self.k, 0.5, 0.5)
        fp_optimal =  1.0 - (1.0 - p_effective)**(1.0 / b)
        
        self.lsh = MinHashLSHBloom(threshold=self.T, num_perm=self.k, fp=fp_optimal, n=self.n, save_dir=self.save_dir)

    def get_minhash(self, text: str, id: int) -> MinHash:
        """
        Compute minhash or grab from file. Minhashes can be
        shared as long as they have the same num_perm.
        """
        # check cache unless explicitly instructed not to
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash:
            if os.path.isfile(mh_pkl):
                with open(mh_pkl, "rb") as f:
                    mh = pickle.load(f)
                assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
                return mh

        # compute minhash
        mh = MinHash(num_perm=self.k)
        assert isinstance(text, str), f"Error empty document with id: {id}"
        
        words = text.split()
        if len(words) < self.ngram:
            s = set(words)
        else:
            s = set([" ".join(words[i:i+self.ngram]) for i in range(len(words) - self.ngram + 1)])
            
        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))
        
        # save to cache
        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)

        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        # compute minhash
        mh = self.get_minhash(text, id)

        # query/insert
        is_dup = self.lsh.query(mh)
        if not is_dup:
            self.lsh.insert(mh)
        
        return is_dup

if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    # if provided, this is an fp experiment
    fp_tag = f"_fp_{args.fp}" if args.fp > 0 else ""
    result_dir = os.path.join(WORK_DIR, benchmark_tag, f"lsh_bloom_results{fp_tag}")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    save_dir = os.path.join(result_dir, f"bloom_filter_{args.sim_threshold}_{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_bloom_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_bloom_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHBloomDeduper(
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



