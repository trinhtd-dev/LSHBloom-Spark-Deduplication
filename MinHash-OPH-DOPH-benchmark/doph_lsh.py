import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_DATA_DIR, ROOT, ngrams, run_lsh_experiment
from oph_lsh import EMPTY, OPHSketcher, hash64


MERSENNE_PRIME = (1 << 61) - 1


def make_2universal(k, seed=42):
    rng = np.random.default_rng(seed)
    a = int(rng.integers(1, MERSENNE_PRIME))
    b = int(rng.integers(0, MERSENNE_PRIME))

    def huniv(x):
        return int(((a * int(x) + b) % MERSENNE_PRIME) % k)

    return huniv


class DOPHSketcher(OPHSketcher):
    """One-permutation hash sketch with deterministic densification."""

    def __init__(self, num_perm):
        super().__init__(num_perm)
        self.huniv = make_2universal(self.num_bins)

    def signature(self, text, doc_id, ngram):
        sig = np.full(self.num_bins, EMPTY, dtype=np.uint64)
        for token in ngrams(text, ngram):
            h = hash64(token)
            bin_id = (h * self.num_bins) >> 64
            if h < int(sig[bin_id]):
                sig[bin_id] = np.uint64(h)

        occupied = np.flatnonzero(sig != EMPTY)
        if len(occupied) == 0:
            raise ValueError(f"empty token set for document {doc_id}")

        occupied_mask = sig != EMPTY
        empty_bins = np.flatnonzero(sig == EMPTY)
        self.last_empty_bins = int(len(empty_bins))
        for i in empty_bins:
            j = self.huniv(int(i))
            if occupied_mask[j]:
                sig[i] = sig[j]
                continue
            for step in range(1, self.num_bins):
                jj = (j + step) % self.num_bins
                if occupied_mask[jj]:
                    sig[i] = sig[jj]
                    break
        return sig


def run(output_root, data_dir, tag, threshold, num_perm, ngram, limit=0, force=False):
    run_lsh_experiment(
        sketch_name="doph",
        sketcher=DOPHSketcher(num_perm),
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
    parser = argparse.ArgumentParser(description="Run MinHashLSH + DOPH.")
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
