import argparse
import hashlib
from pathlib import Path

import numpy as np
from datasketch import MinHash

from common import DEFAULT_DATA_DIR, ROOT, ngrams, run_lsh_experiment


EMPTY = np.uint64((1 << 64) - 1)


def hash64(value, person=b"ophbench"):
    digest = hashlib.blake2b(value.encode("utf8"), digest_size=8, person=person)
    return int.from_bytes(digest.digest(), "big", signed=False)


def mix64(*values):
    digest = hashlib.blake2b(digest_size=8, person=b"ophmix")
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return np.uint64(int.from_bytes(digest.digest(), "big", signed=False))


def signature_to_minhash(signature):
    mh = MinHash(num_perm=len(signature))
    mh.hashvalues = (signature >> np.uint64(32)).astype(np.uint64)
    return mh


def hashvalues_to_minhash(hashvalues):
    mh = MinHash(num_perm=len(hashvalues))
    mh.hashvalues = np.asarray(hashvalues, dtype=np.uint64)
    return mh


class OPHSketcher:
    """One-permutation hash sketch adapted to fixed-length MinHashLSH.

    Empty bins are filled with document-id salted values to avoid common
    empty-bin collisions. This is a non-densified OPH baseline adapter.
    """

    cache_ext = ".npy"

    def __init__(self, num_perm):
        self.num_bins = int(num_perm)
        self.last_empty_bins = 0

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

        empty_bins = np.flatnonzero(sig == EMPTY)
        self.last_empty_bins = int(len(empty_bins))
        for i in empty_bins:
            sig[i] = mix64(i, hash64(str(doc_id), person=b"ophdoc"))
        return sig

    def to_minhash(self, text, doc_id, ngram):
        return signature_to_minhash(self.signature(text, doc_id, ngram))

    def save(self, path, mh):
        np.save(path, mh.hashvalues)

    def load(self, path):
        return hashvalues_to_minhash(np.load(path))


def run(output_root, data_dir, tag, threshold, num_perm, ngram, limit=0, force=False):
    run_lsh_experiment(
        sketch_name="oph",
        sketcher=OPHSketcher(num_perm),
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
    parser = argparse.ArgumentParser(description="Run MinHashLSH + OPH.")
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
