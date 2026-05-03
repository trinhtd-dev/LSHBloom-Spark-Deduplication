import argparse
import os
import pickle
import random
import sys
from typing import Sequence

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../synthetic_benchmark")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash, MinHashLSH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-threshold", default=0.8)
    parser.add_argument("--num-perm", default=128)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument(
        "--probe-radius",
        type=int,
        default=2,
        help="How many neighboring bucket variants to probe per band.",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=42,
        help="Seed for reproducible multi-probe offsets.",
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
        help="Force recomputing minhashes instead of using cached values.",
    )
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


class MultiProbeLSH(MinHashLSH):
    """MinHashLSH with a standard multi-probe expansion over band variants."""

    def __init__(self, *args, probe_radius: int = 2, probe_seed: int = 42, max_probes: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_radius = int(probe_radius)
        self.probe_seed = int(probe_seed)
        self.max_probes = int(max_probes)

    def _perturb_band(self, band):
        yield list(band)
        for i in range(len(band)):
            for delta in range(1, self.probe_radius + 1):
                for sign in (-1, 1):
                    new_band = list(band)
                    new_band[i] = (int(new_band[i]) + sign * delta) & ((1 << 64) - 1)
                    yield new_band

    def query(self, minhash) -> list:
        if len(minhash) != self.h:
            raise ValueError(
                "Expecting minhash with length %d, got %d" % (self.h, len(minhash))
            )

        candidates = set(super().query(minhash))
        seen = set()
        probes = 0

        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            band = minhash.hashvalues[start:end]
            for variant in self._perturb_band(band):
                key = tuple(int(v) for v in variant)
                if key in seen:
                    continue
                seen.add(key)

                h = self._H(np.asarray(variant, dtype=np.uint64))
                if h in hashtable:
                    candidates.update(hashtable[h])

                probes += 1
                if probes >= self.max_probes:
                    return list(candidates)
        return list(candidates)


class LSHMultiProbeDeduper(DedupHarness):
    def __init__(self, sim_threshold, num_perm, minhash_root, recompute_minhashes=False, ngram=1, probe_radius=2, probe_seed=42):
        super().__init__("lsh_multiprobe")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = int(ngram)
        self.lsh = MultiProbeLSH(
            threshold=self.T,
            num_perm=self.k,
            storage_config={"type": "dict"},
            probe_radius=probe_radius,
            probe_seed=probe_seed,
        )

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash and os.path.isfile(mh_pkl):
            with open(mh_pkl, "rb") as f:
                mh = pickle.load(f)
            if isinstance(mh, MinHash):
                return mh

        mh = MinHash(num_perm=self.k)
        assert isinstance(text, str), f"Error empty document with id: {id}"
        words = text.split()
        if len(words) < self.ngram:
            s = set(words)
        else:
            s = set([" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)])
        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))

        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        is_dup = bool(self.lsh.query(mh))
        if not is_dup:
            self.lsh.insert(id, mh)
        return is_dup


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_multiprobe_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_multiprobe_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_multiprobe_{args.sim_threshold}_{args.num_perm}_score.csv")

    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHMultiProbeDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        probe_radius=int(args.probe_radius),
        probe_seed=int(args.probe_seed),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
