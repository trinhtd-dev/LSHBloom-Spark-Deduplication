import argparse
import pickle
from pathlib import Path

from datasketch import MinHash

from common import DEFAULT_DATA_DIR, ROOT, ngrams, run_lsh_experiment


class StandardMinHashSketcher:
    """Baseline from LSH-benchmark/dedup/lsh/lsh.py."""

    cache_ext = ".pkl"

    def __init__(self, num_perm):
        self.num_perm = int(num_perm)

    def to_minhash(self, text, doc_id, ngram):
        tokens = ngrams(text, ngram)
        if not tokens:
            raise ValueError(f"empty token set for document {doc_id}")
        mh = MinHash(num_perm=self.num_perm)
        for token in tokens:
            mh.update(token.encode("utf8"))
        return mh

    def save(self, path, mh):
        with open(path, "wb") as fout:
            pickle.dump(mh, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as fin:
            return pickle.load(fin)


def run(output_root, data_dir, tag, threshold, num_perm, ngram, limit=0, force=False):
    run_lsh_experiment(
        sketch_name="minhash",
        sketcher=StandardMinHashSketcher(num_perm),
        output_root=output_root,
        data_dir=data_dir,
        tag=tag,
        threshold=threshold,
        num_perm=num_perm,
        ngram=ngram,
        limit=limit,
        force=force,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard MinHashLSH baseline.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(ROOT / "runs"))
    parser.add_argument("--input", default="test_p_0.1")
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        output_root=Path(args.output_root),
        data_dir=Path(args.data_dir),
        tag=args.input,
        threshold=args.sim_threshold,
        num_perm=args.num_perm,
        ngram=args.ngram,
        limit=args.limit,
        force=args.force,
    )
