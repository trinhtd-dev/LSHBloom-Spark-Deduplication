import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from config import *
from dedup_parsing_harness import DedupHarness
from cc_net.dedup import *
from cc_net.flat_hash_set import HASH_TYPE, AbstractDedupHashSet, FlatHashSet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--sim-threshold",
		help="Text overlap threshold for deduplication, should be in [0, 1]. Default is 0.8",
		default=0.8,
	)
    parser.add_argument(
        "--input",
        help="Input tag",
        type=str,
        required=True
    )
    return parser.parse_args()

class CCNetDeduper(DedupHarness):
    def __init__(self, sim_threshold: float, save_dir: str):
        super().__init__(f"ccnet_{sim_threshold}")
        self.T = sim_threshold
        self.save_dir = save_dir
        self.hashset = FlatHashSet()
        
    def teardown(self):
        self.hashset.dump(os.path.join(self.save_dir, "hashset.npy"))

    def deduplicate(self, text: str, id: int) -> bool:
        hash = compute_hashes(text)
        # insertion into the hashset yields duplicate info for paragraphs
        dups = self.hashset.add(hash)
        # compare dup portions against threshold
        p = dups.sum() / len(dups)
        is_dup = p >= self.T
        return is_dup

if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    run_id = f"{args.sim_threshold}"
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "ccnet_results")
    output_file = os.path.join(result_dir, f"ccnet_{run_id}_preds.csv")
    result_file = os.path.join(result_dir, f"ccnet_{run_id}_score.csv")
    save_dir = os.path.join(result_dir, f"ccnet_{run_id}_hashes")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    deduper = CCNetDeduper(sim_threshold=float(args.sim_threshold), save_dir=save_dir)
    
    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)



