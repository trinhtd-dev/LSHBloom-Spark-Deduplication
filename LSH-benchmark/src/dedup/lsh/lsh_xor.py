import argparse
import hashlib
import os
import pickle
import sys
import time
from glob import glob
from typing import Iterable, List

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../../synthetic_benchmark")))

from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash
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
        "--ngram",
        help="N-gram size for MinHashing. Default is 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fingerprint-bits",
        help="Fingerprint size for XOR filter (8 or 16). Default is 8",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rebuild-every",
        help="Rebuild XOR filter after this many new band keys. Default is 5000",
        type=int,
        default=5000,
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


class XorFilter:
    """
    Xor filter with 3 hash locations and fixed-size fingerprints.
    Construction is static; we rebuild periodically for streaming use.
    """

    def __init__(self, fingerprint_bits: int = 8, max_tries: int = 20):
        if fingerprint_bits not in (8, 16):
            raise ValueError("fingerprint_bits must be 8 or 16")
        self.fingerprint_bits = fingerprint_bits
        self.max_tries = max_tries
        self.block_length = 0
        self.seed = 0
        self.fingerprints: List[int] = []

    def _splitmix64(self, x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def _hash64(self, key: int, seed: int) -> int:
        return self._splitmix64(key ^ seed)

    def _fingerprint(self, h: int) -> int:
        mask = (1 << self.fingerprint_bits) - 1
        return h & mask

    def _positions(self, h: int, block_length: int):
        h1 = h % block_length
        h2 = ((h >> 21) % block_length) + block_length
        h3 = ((h >> 42) % block_length) + 2 * block_length
        return int(h1), int(h2), int(h3)

    def build(self, keys: Iterable[int]) -> bool:
        keys_list = list(keys)
        if not keys_list:
            self.block_length = 0
            self.seed = 0
            self.fingerprints = []
            return True

        n = len(keys_list)
        size = max(3, int(n * 1.23))
        block_length = max(1, size // 3)

        for attempt in range(self.max_tries):
            seed = 0xA5A5A5A5 + attempt
            counts = [0] * (3 * block_length)
            xors = [0] * (3 * block_length)

            for key in keys_list:
                h = self._hash64(key, seed)
                h1, h2, h3 = self._positions(h, block_length)
                counts[h1] += 1
                counts[h2] += 1
                counts[h3] += 1
                xors[h1] ^= h
                xors[h2] ^= h
                xors[h3] ^= h

            stack = [i for i, c in enumerate(counts) if c == 1]
            order = []

            while stack:
                i = stack.pop()
                if counts[i] == 0:
                    continue
                h = xors[i]
                order.append((i, h))
                h1, h2, h3 = self._positions(h, block_length)
                for pos in (h1, h2, h3):
                    counts[pos] -= 1
                    xors[pos] ^= h
                    if counts[pos] == 1:
                        stack.append(pos)

            if len(order) != n:
                continue

            fingerprints = [0] * (3 * block_length)
            for i, h in reversed(order):
                fp = self._fingerprint(h)
                h1, h2, h3 = self._positions(h, block_length)
                fingerprints[i] = fp ^ fingerprints[h1] ^ fingerprints[h2] ^ fingerprints[h3]

            self.block_length = block_length
            self.seed = seed
            self.fingerprints = fingerprints
            return True

        return False

    def contains(self, key: int) -> bool:
        if not self.fingerprints:
            return False
        h = self._hash64(key, self.seed)
        h1, h2, h3 = self._positions(h, self.block_length)
        fp = self._fingerprint(h)
        return fp == (self.fingerprints[h1] ^ self.fingerprints[h2] ^ self.fingerprints[h3])


class LSHXorDeduper(DedupHarness):
    def __init__(
        self,
        n: int,
        sim_threshold: float,
        num_perm: int,
        minhash_root: str,
        save_dir: str,
        recompute_minhashes: bool = False,
        ngram: int = 1,
        fingerprint_bits: int = 8,
        rebuild_every: int = 5000,
        cache_minhash: bool = True,
    ):
        super().__init__("lsh_xor")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.n = int(n)
        self.save_dir = save_dir
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.cache_minhash = cache_minhash
        self.ngram = int(ngram)
        self.rebuild_every = int(rebuild_every)

        self.bands, self.rows = _optimal_param(self.T, self.k, 0.5, 0.5)
        self._filter = XorFilter(fingerprint_bits=fingerprint_bits)
        self._keys = set()
        self._pending = set()

        for item in glob(f"{save_dir}/*.xor"):
            os.remove(item)
            print(f"Clearing XOR filter cache: {item}")

    def _band_key(self, band_index: int, band_slice) -> int:
        digest = hashlib.blake2b(
            band_slice.tobytes() + band_index.to_bytes(4, "little"),
            digest_size=8,
            person=b"lsh-xor",
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _band_keys(self, mh: MinHash) -> List[int]:
        keys = []
        total = self.bands * self.rows
        if total == 0:
            return keys
        for band in range(self.bands):
            start = band * self.rows
            end = start + self.rows
            if end > len(mh.hashvalues):
                break
            keys.append(self._band_key(band, mh.hashvalues[start:end]))
        return keys

    def _maybe_rebuild(self) -> float:
        if len(self._pending) < self.rebuild_every:
            return 0.0
        self._keys.update(self._pending)
        self._pending.clear()
        t0 = time.perf_counter()
        built = self._filter.build(self._keys)
        elapsed = time.perf_counter() - t0
        if not built:
            print("[WARNING] XOR filter build failed; falling back to exact set checks only.")
            self._filter = XorFilter(fingerprint_bits=self._filter.fingerprint_bits)
        return elapsed

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if self.cache_minhash and not self.force_minhash:
            if os.path.isfile(mh_pkl):
                with open(mh_pkl, "rb") as f:
                    mh = pickle.load(f)
                assert isinstance(mh, MinHash), f"Failed to parse minhash at: {mh_pkl}"
                return mh

        mh = MinHash(num_perm=self.k)
        assert isinstance(text, str), f"Error empty document with id: {id}"

        words = text.split()
        if len(words) < self.ngram:
            s = set(words)
        else:
            s = set(" ".join(words[i:i + self.ngram]) for i in range(len(words) - self.ngram + 1))

        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))

        if self.cache_minhash:
            with open(mh_pkl, "wb") as f:
                pickle.dump(mh, f)

        return mh

    def deduplicate_minhash(self, mh: MinHash) -> tuple[bool, float, float]:
        t0 = time.perf_counter()
        keys = self._band_keys(mh)
        if not keys:
            return False, 0.0, 0.0

        query_sec = time.perf_counter() - t0

        is_dup = False
        for key in keys:
            if self._filter.contains(key) or key in self._pending:
                is_dup = True
                break

        insert_sec = 0.0
        if not is_dup:
            for key in keys:
                self._pending.add(key)
            insert_sec = self._maybe_rebuild()

        return is_dup, query_sec, insert_sec

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        is_dup, _, _ = self.deduplicate_minhash(mh)
        return is_dup

        return is_dup

    def teardown(self) -> None:
        if self._pending:
            self._keys.update(self._pending)
            self._pending.clear()
            self._filter.build(self._keys)

        filter_path = os.path.join(self.save_dir, "xor_filter_state.xor")
        if self._filter.fingerprints:
            with open(filter_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint_bits": self._filter.fingerprint_bits,
                        "block_length": self._filter.block_length,
                        "seed": self._filter.seed,
                        "fingerprints": self._filter.fingerprints,
                    },
                    f,
                )


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_xor_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    save_dir = os.path.join(result_dir, f"xor_filter_{args.sim_threshold}_{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_xor_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_xor_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHXorDeduper(
        n=DATA_SIZE,
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        save_dir=save_dir,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        fingerprint_bits=int(args.fingerprint_bits),
        rebuild_every=int(args.rebuild_every),
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
