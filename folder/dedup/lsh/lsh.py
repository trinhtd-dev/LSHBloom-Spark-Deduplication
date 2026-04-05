import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSH
import pickle

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
		"--redis-port",
		help="Port for redis index. Default is 6379.",
        type=int,
		default=6379,
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

class LSHDeduper(DedupHarness):
    def __init__(self, sim_threshold: float, num_perm: int, minhash_root: str, redis_params: dict, recompute_minhashes=False, ngram: int=1):
        super().__init__("lsh")
        self.T = sim_threshold
        self.k = num_perm
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram
        
        self.lsh = MinHashLSH(threshold=self.T, num_perm=self.k, storage_config=redis_params)

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
        query_result = self.lsh.query(mh)
        uniq = not len(query_result) or (len(query_result) == 1 and query_result[0] == id)
        is_dup = not uniq
        if not is_dup:
            self.lsh.insert(id, mh)
        
        return is_dup

if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    import time
    redis_params =  {
            "type": "redis",
            "basename": f"tpc_{benchmark_tag}_{int(time.time())}".encode('utf-8'),
            "redis": {"host": "localhost", "port": args.redis_port},
        }

    deduper = LSHDeduper(
        sim_threshold=float(args.sim_threshold), 
        num_perm=int(args.num_perm), 
        minhash_root=minhash_root,
        redis_params=redis_params,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram)
        )
    
    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)



