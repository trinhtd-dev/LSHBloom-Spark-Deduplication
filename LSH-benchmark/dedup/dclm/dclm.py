import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from glob import glob
import argparse
from config import *
from dedup_parsing_harness import DedupHarness
from pybloomfilter import BloomFilter
from dclm_deduper import process_document, tokenize_doc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--sim-threshold",
		help="Text overlap threshold for deduplication, should be in [0, 1]. Default is 0.8",
		default=0.8,
	)
    parser.add_argument(
		"--ngram-size",
		help="Size of ngrams (ngram mode only). Default is 13",
		default=13,
	)
    parser.add_argument(
        "--input",
        help="Input tag",
        type=str,
        required=True
    )
    return parser.parse_args()

class DclmDeduper(DedupHarness):
    def __init__(self, n: int, sim_threshold: float, ngram_size: int, save_dir: str):
        name = f"dclm_{sim_threshold}_{ngram_size}"
        super().__init__(name)
        self.T = sim_threshold
        self.ngram_size = ngram_size
        self.n = n
        self.save_file = os.path.join(save_dir, f"{name}.bf")
        
        # clear save_dir if re-running
        for item in glob(f"{save_dir}/*.bf"):
            os.remove(item)
            print(f"Clearing bloom filter: {item}")
        
        #                           n, fp, save_file
        self.bf = BloomFilter(self.n, 0.00001, self.save_file)

    def deduplicate(self, text: str, id: int) -> bool:
        paragraphs, total_ngrams = tokenize_doc(text, self.ngram_size)
        num_dup_ngrams = process_document(paragraphs, self.bf)
        if not total_ngrams:
            return False
        is_dup = (num_dup_ngrams / total_ngrams) >= self.T
        return is_dup

if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    name = f"dclm"
    run_id = f"{args.sim_threshold}_{args.ngram_size}"
    result_dir = os.path.join(WORK_DIR, benchmark_tag, f"{name}_results")
    save_dir = os.path.join(result_dir, f"bloom_filter_{run_id}")
    output_file = os.path.join(result_dir, f"{name}_{run_id}_preds.csv")
    result_file = os.path.join(result_dir, f"{name}_{run_id}_score.csv")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(WORK_DIR, benchmark_tag, f"ngram_count_{args.ngram_size}.txt"), 'r') as f:
        N = int(f.readline())

    deduper = DclmDeduper(
        n=N, 
        sim_threshold=float(args.sim_threshold),
        ngram_size=int(args.ngram_size),
        save_dir=save_dir
        )
    
    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)



