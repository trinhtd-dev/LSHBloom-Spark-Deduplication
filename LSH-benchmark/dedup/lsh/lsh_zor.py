import argparse
import hashlib
import os
import pickle
import sys
import time
from glob import glob
from typing import Dict, Iterable, List, Set

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../synthetic_benchmark")))

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
        help="Fingerprint size for ZOR filter (8 or 16). Default is 8",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--spill-structure",
        help="Auxiliary spill structure type for unresolved keys",
        choices=["exact", "set"],
        default="exact",
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


class ZorFilter:
    """
    ZOR-style filter for streaming-friendly dedup.

    Main path keeps XOR-style 3-position fingerprint query.
    Unresolved keys from deterministic peeling are stored in a small auxiliary spill set.
    """

    def __init__(self, fingerprint_bits: int = 8, max_tries: int = 1):
        if fingerprint_bits not in (8, 16):
            raise ValueError("fingerprint_bits must be 8 or 16")
        self.fingerprint_bits = fingerprint_bits
        self.max_tries = max_tries
        self.block_length = 0
        self.seed = 0
        self.fingerprints: List[int] = []
        self.spill_keys: set[int] = set()

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
        fp = h & mask
        return fp if fp != 0 else 1

    def _positions(self, h: int, block_length: int):
        h1 = h % block_length
        h2 = ((h >> 21) % block_length) + block_length
        h3 = ((h >> 42) % block_length) + 2 * block_length
        return int(h1), int(h2), int(h3)

    def build(self, keys: Iterable[int]) -> bool:
        keys_list = list(dict.fromkeys(keys))
        self.spill_keys = set()
        if not keys_list:
            self.block_length = 0
            self.seed = 0
            self.fingerprints = []
            return True

        n = len(keys_list)
        size = max(3, int(n * 1.23))
        block_length = max(1, size // 3)
        seed = 0xC0FFEE00

        counts = [0] * (3 * block_length)
        xors = [0] * (3 * block_length)
        incident: List[List[int]] = [[] for _ in range(3 * block_length)]
        key_hashes: list[tuple[int, int, int, int]] = []

        for key in keys_list:
            h = self._hash64(key, seed)
            h1, h2, h3 = self._positions(h, block_length)
            key_hashes.append((key, h, h1, h2, h3))
            for pos in (h1, h2, h3):
                counts[pos] += 1
                xors[pos] ^= h
                incident[pos].append(key)

        stack = [i for i, c in enumerate(counts) if c == 1]
        order = []
        removed = set()

        while stack:
            i = stack.pop()
            if counts[i] == 0:
                continue
            h = xors[i]
            order.append((i, h))
            removed.add(h)
            h1, h2, h3 = self._positions(h, block_length)
            for pos in (h1, h2, h3):
                counts[pos] -= 1
                xors[pos] ^= h
                if counts[pos] == 1:
                    stack.append(pos)

        unresolved_keys = []
        if len(removed) != n:
            for key, h, h1, h2, h3 in key_hashes:
                if h not in removed:
                    unresolved_keys.append(key)

        fingerprints = [0] * (3 * block_length)
        for i, h in reversed(order):
            fp = self._fingerprint(h)
            h1, h2, h3 = self._positions(h, block_length)
            fingerprints[i] = fp ^ fingerprints[h1] ^ fingerprints[h2] ^ fingerprints[h3]

        self.block_length = block_length
        self.seed = seed
        self.fingerprints = fingerprints
        self.spill_keys = set(unresolved_keys)
        return True

    def contains(self, key: int) -> bool:
        if not self.fingerprints:
            return key in self.spill_keys
        if key in self.spill_keys:
            return True
        h = self._hash64(key, self.seed)
        h1, h2, h3 = self._positions(h, self.block_length)
        fp = self._fingerprint(h)
        return fp == (self.fingerprints[h1] ^ self.fingerprints[h2] ^ self.fingerprints[h3])


class LSHZorDeduper(DedupHarness):
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
        spill_structure: str = "exact",
        rebuild_every: int = 5000,
        cache_minhash: bool = True,
    ):
        super().__init__("lsh_zor")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.n = int(n)
        self.save_dir = save_dir
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.cache_minhash = cache_minhash
        self.ngram = int(ngram)
        self.spill_structure = spill_structure
        self.rebuild_every = int(rebuild_every)

        self.bands, self.rows = _optimal_param(self.T, self.k, 0.5, 0.5)
        self._filter = ZorFilter(fingerprint_bits=fingerprint_bits)
        self._keys: Set[int] = set()
        self._pending: Set[int] = set()

        # Verification store: ensure candidate from LSH is verified by estimated Jaccard.
        # Without this step, pipeline is only candidate generation and can over-predict duplicates.
        self._doc_minhash: Dict[int, MinHash] = {}
        self._band_to_docs: Dict[int, Set[int]] = {}

        os.makedirs(save_dir, exist_ok=True)
        for item in glob(f"{save_dir}/*.zor"):
            os.remove(item)
            print(f"Clearing ZOR filter cache: {item}")

    def _band_key(self, band_index: int, band_slice) -> int:
        digest = hashlib.blake2b(
            band_slice.tobytes() + band_index.to_bytes(4, "little"),
            digest_size=8,
            person=b"lsh-zor",
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

    def _sync_filter(self) -> float:
        if len(self._pending) < self.rebuild_every:
            return 0.0
        self._keys.update(self._pending)
        self._pending.clear()
        t0 = time.perf_counter()
        self._filter.build(self._keys)
        return time.perf_counter() - t0

    def deduplicate_minhash(self, mh: MinHash, doc_id: int) -> tuple[bool, float, float]:
        t0 = time.perf_counter()
        keys = self._band_keys(mh)
        if not keys:
            # still index the doc for future comparisons
            self._doc_minhash[doc_id] = mh
            return False, 0.0, 0.0

        # Candidate generation by LSH bands + ZOR/pending gate
        candidate_ids: Set[int] = set()
        maybe_candidate = False
        for key in keys:
            if self._filter.contains(key) or key in self._pending or key in self._band_to_docs:
                maybe_candidate = True
            docs = self._band_to_docs.get(key)
            if docs:
                candidate_ids.update(docs)

        # Verification step: confirm by MinHash-estimated Jaccard threshold
        is_dup = False
        if maybe_candidate and candidate_ids:
            for cid in candidate_ids:
                cand_mh = self._doc_minhash.get(cid)
                if cand_mh is None:
                    continue
                if mh.jaccard(cand_mh) >= self.T:
                    is_dup = True
                    break

        query_sec = time.perf_counter() - t0

        insert_sec = 0.0
        if not is_dup:
            for key in keys:
                self._band_to_docs.setdefault(key, set()).add(doc_id)
            self._doc_minhash[doc_id] = mh
            self._pending.update(keys)
            insert_sec = self._sync_filter()

        return is_dup, query_sec, insert_sec

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        is_dup, _, _ = self.deduplicate_minhash(mh, id)
        return is_dup

    def teardown(self) -> None:
        if self._pending:
            self._keys.update(self._pending)
            self._pending.clear()
            self._filter.build(self._keys)

        filter_path = os.path.join(self.save_dir, "zor_filter_state.zor")
        if self._filter.fingerprints or self._filter.spill_keys:
            with open(filter_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint_bits": self._filter.fingerprint_bits,
                        "block_length": self._filter.block_length,
                        "seed": self._filter.seed,
                        "fingerprints": self._filter.fingerprints,
                        "spill_keys": list(self._filter.spill_keys),
                    },
                    f,
                )


if __name__ == "__main__":
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    result_dir = os.path.join(WORK_DIR, benchmark_tag, "lsh_zor_results")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    save_dir = os.path.join(result_dir, f"zor_filter_{args.sim_threshold}_{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_zor_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_zor_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHZorDeduper(
        n=DATA_SIZE,
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        save_dir=save_dir,
        recompute_minhashes=args.force_compute_minhash,
        ngram=int(args.ngram),
        fingerprint_bits=int(args.fingerprint_bits),
        spill_structure=args.spill_structure,
    )

    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
