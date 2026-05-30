import os
import sys
import math
import pickle
import numpy as np
import mmh3
import argparse
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "datasketch")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "../../../synthetic_benchmark")))

from glob import glob
from config import *
from dedup_parsing_harness import DedupHarness
from datasketch import MinHash
from scipy.integrate import quad as integrate

FP_DEFAULT = 1e-5


class BlockedBloomFilter:
    """
    Bloom filter với cache-line aligned blocks (512 bits = 64 bytes mỗi block).
    Mỗi key chỉ truy cập đúng 1 block → 1 cache miss thay vì k cache misses.
    """
    BLOCK_SIZE = 512

    def __init__(self, n: int, fp_rate: float = 1e-10):
        self.k = max(1, int(-math.log(fp_rate) / math.log(2)))
        m_bits = int(-n * math.log(fp_rate) / (math.log(2) ** 2))
        correction = self._correction_factor(self.k)
        m_bits = int(m_bits * correction)
        num_blocks = max(1, (m_bits + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE)
        self.num_blocks = num_blocks
        self.m = num_blocks * self.BLOCK_SIZE
        self.blocks = np.zeros((num_blocks, 8), dtype=np.uint64)
        self.fp_rate = fp_rate
        self.n = n

    @staticmethod
    def _correction_factor(k: int) -> float:
        table = {
            1: 1.0, 2: 1.05, 3: 1.1, 4: 1.15, 5: 1.2,
            6: 1.25, 7: 1.3, 8: 1.4, 9: 1.55, 10: 1.74,
            11: 2.0, 12: 2.3, 13: 2.7, 14: 3.2, 15: 4.0,
            16: 5.5, 17: 8.0
        }
        return table.get(k, 2.0)

    def _get_block_and_bits(self, item: int):
        item_bytes = item.to_bytes(8, byteorder='little', signed=False)
        block_idx = mmh3.hash(item_bytes, seed=0, signed=False) % self.num_blocks
        bit_positions = [
            mmh3.hash(item_bytes, seed=i, signed=False) % self.BLOCK_SIZE
            for i in range(1, self.k + 1)
        ]
        return block_idx, bit_positions

    def add(self, item: int):
        block_idx, bit_positions = self._get_block_and_bits(item)
        bit_pos_arr = np.array(bit_positions, dtype=np.int64)
        word_indices = bit_pos_arr // 64
        bit_offsets = bit_pos_arr % 64
        masks = np.uint64(1) << bit_offsets.astype(np.uint64)
        np.bitwise_or.at(self.blocks[block_idx], word_indices, masks)

    def contains(self, item: int) -> bool:
        block_idx, bit_positions = self._get_block_and_bits(item)
        bit_pos_arr = np.array(bit_positions, dtype=np.int64)
        word_indices = bit_pos_arr // 64
        bit_offsets = bit_pos_arr % 64
        masks = np.uint64(1) << bit_offsets.astype(np.uint64)
        return bool(np.all(self.blocks[block_idx][word_indices] & masks))

    def __contains__(self, item: int) -> bool:
        return self.contains(item)

    def save(self, path: str):
        np.save(path, self.blocks)

    @classmethod
    def load(cls, path: str, n: int, fp_rate: float):
        bbf = cls(n, fp_rate)
        actual_path = path if path.endswith(".npy") else path + ".npy"
        if os.path.exists(actual_path):
            bbf.blocks = np.load(actual_path)
        return bbf


class BlockedBloomTable:
    def __init__(self, item_count: int, fp: float, num_arrays: int, fname: str = None, max_size: int = None):
        self.r = num_arrays
        self.fname = fname
        if max_size is not None and item_count > max_size:
            item_count = max_size
        npy_exists = fname is not None and os.path.exists(fname + ".npy")
        if fname is not None and (os.path.exists(fname) or npy_exists):
            print(f"Loading Blocked Bloom Filter at {fname}...")
            self.bloom_filter = BlockedBloomFilter.load(fname, item_count, fp)
        else:
            self.bloom_filter = BlockedBloomFilter(n=item_count, fp_rate=fp)

    def sync(self):
        if self.fname is not None:
            self.bloom_filter.save(self.fname)

    def assert_size(self, hashvalues: List[int]):
        if not len(hashvalues) == self.r:
            raise RuntimeError(f"Invalid length for indices, {len(hashvalues)}, expected {self.r} items")

    def hash(self, hashvalues: List[int]) -> int:
        self.assert_size(hashvalues)
        arr = np.array(hashvalues, dtype=np.uint64)
        return mmh3.hash(arr.tobytes(), seed=42, signed=False)

    def insert(self, hashvalues: List[int]) -> None:
        x = self.hash(hashvalues)
        self.bloom_filter.add(x)

    def query(self, hashvalues: List[int]) -> bool:
        x = self.hash(hashvalues)
        return x in self.bloom_filter


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
            fp, pe = _false_positive_probability(threshold, b, r)
            fn, ne = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSHBlockedBloom(object):
    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        n: int = None,
        fp: float = None,
        save_dir: str = None,
    ) -> None:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        false_positive_weight, false_negative_weight = weights
        self.b, self.r = _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.hashtables = [
            BlockedBloomTable(
                item_count=n,
                fp=fp,
                num_arrays=self.r,
                fname=os.path.join(save_dir, f"band-{i}.bf") if save_dir is not None else None,
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

    def insert(self, minhash: MinHash):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        Hs = [minhash.hashvalues[start:end] for start, end in self.hashranges]
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H)

    def query(self, minhash) -> bool:
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = minhash.hashvalues[start:end]
            if hashtable.query(H):
                return True
        return False

    def sync(self):
        print("Saving Blocked Bloom Index...")
        for table in self.hashtables:
            table.sync()


class LSHBlockedBloomDeduper(DedupHarness):
    def __init__(self, n: int, sim_threshold: float, num_perm: int, minhash_root: str, save_dir: str,
                 recompute_minhashes=False, fp=FP_DEFAULT, ngram: int = 1, clear_filter_state: bool = False):
        super().__init__("lsh_blocked_bloom")
        self.T = sim_threshold
        self.k = num_perm
        self.n = n
        self.save_dir = save_dir
        self.minhash_root = minhash_root
        self.force_minhash = recompute_minhashes
        self.ngram = ngram
        if clear_filter_state:
            for item in glob(f"{save_dir}/*.bf.npy"):
                os.remove(item)
                print(f"Clearing blocked bloom filter: {item}")
        p_effective = fp
        b, r = _optimal_param(self.T, self.k, 0.5, 0.5)
        fp_optimal = 1.0 - (1.0 - p_effective) ** (1.0 / b)
        self.lsh = MinHashLSHBlockedBloom(threshold=self.T, num_perm=self.k, fp=fp_optimal, n=self.n, save_dir=self.save_dir)

    def get_minhash(self, text: str, id: int) -> MinHash:
        mh_pkl = os.path.join(self.minhash_root, f"{id}.pkl")
        if not self.force_minhash:
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
            s = set([" ".join(words[i:i + self.ngram]) for i in range(len(words) - self.ngram + 1)])
        assert len(s) > 0, f"Error: empty document with id: {id}"
        for d in s:
            mh.update(d.encode("utf8"))
        with open(mh_pkl, "wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        mh = self.get_minhash(text, id)
        is_dup = self.lsh.query(mh)
        if not is_dup:
            self.lsh.insert(mh)
        return is_dup

    def teardown(self) -> None:
        self.lsh.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-threshold", default=0.8, type=float)
    parser.add_argument("--num-perm", default=128, type=int)
    parser.add_argument("--fp", type=float, default=FP_DEFAULT)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--force-compute-minhash", action='store_true')
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    benchmark_tag = args.input
    benchmark_csv = os.path.join(DATA_PATH, f"{benchmark_tag}.csv")
    benchmark_jsonl = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
    fp_tag = f"_fp_{args.fp}" if args.fp != FP_DEFAULT else ""
    result_dir = os.path.join(WORK_DIR, benchmark_tag, f"lsh_blocked_bloom_results{fp_tag}")
    minhash_root = os.path.join(result_dir, "minhashes", f"{args.num_perm}")
    save_dir = os.path.join(result_dir, f"blocked_bloom_filter_{args.sim_threshold}_{args.num_perm}")
    output_file = os.path.join(result_dir, f"lsh_blocked_bloom_{args.sim_threshold}_{args.num_perm}_preds.csv")
    result_file = os.path.join(result_dir, f"lsh_blocked_bloom_{args.sim_threshold}_{args.num_perm}_score.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    deduper = LSHBlockedBloomDeduper(
        n=DATA_SIZE,
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        minhash_root=minhash_root,
        save_dir=save_dir,
        recompute_minhashes=args.force_compute_minhash,
        fp=args.fp,
        ngram=int(args.ngram)
    )
    deduper.run(benchmark_jsonl, output_file)
    deduper.score(output_file, benchmark_csv, result_file)
