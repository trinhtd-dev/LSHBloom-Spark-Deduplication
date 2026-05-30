from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

from tqdm.autonotebook import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
BENCHMARK_ROOT = REPO_ROOT / "LSH-benchmark"

sys.path.insert(0, str(BENCHMARK_ROOT / "dedup"))
sys.path.insert(0, str(BENCHMARK_ROOT / "dedup" / "lsh" / "datasketch"))
sys.path.insert(0, str(BENCHMARK_ROOT / "synthetic_benchmark"))
sys.path.insert(0, str(CURRENT_DIR))

from config import DATA_PATH, DATA_SIZE, WORK_DIR  # type: ignore  # noqa: E402
from dedup_parsing_harness import DedupHarness  # type: ignore  # noqa: E402
from datasketch import MinHash, MinHashLSH  # type: ignore  # noqa: E402
from minhash_lsh_cpp import scoped_text  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard MinHashLSH over char shingles.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=150)
    parser.add_argument("--force-compute-minhash", action="store_true")
    return parser.parse_args()


class CharMinHashLSHDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        shingle_size: int,
        max_shingles: int,
        minhash_root: Path,
        recompute_minhashes: bool = False,
    ):
        super().__init__("char_minhash_lsh")
        self.T = float(sim_threshold)
        self.k = int(num_perm)
        self.shingle_size = int(shingle_size)
        self.max_shingles = int(max_shingles)
        self.minhash_root = minhash_root
        self.force_minhash = bool(recompute_minhashes)
        self.lsh = MinHashLSH(threshold=self.T, num_perm=self.k, storage_config={"type": "dict"})
        self.minhash_root.mkdir(parents=True, exist_ok=True)

    def get_minhash(self, text: str, doc_id: str) -> MinHash | None:
        mh_pkl = self.minhash_root / f"{doc_id}.pkl"
        if not self.force_minhash and mh_pkl.exists():
            with mh_pkl.open("rb") as f:
                mh = pickle.load(f)
            return mh

        scoped = scoped_text(text, self.shingle_size, self.max_shingles)
        if len(scoped) < self.shingle_size:
            return None

        mh = MinHash(num_perm=self.k)
        for i in range(len(scoped) - self.shingle_size + 1):
            mh.update(scoped[i : i + self.shingle_size].encode("utf-8"))

        with mh_pkl.open("wb") as f:
            pickle.dump(mh, f)
        return mh

    def deduplicate(self, text: str, id: int) -> bool:
        doc_id = str(id)
        mh = self.get_minhash(text, doc_id)
        if mh is None:
            return False

        query_result = self.lsh.query(mh)
        is_dup = bool(query_result)
        if not is_dup:
            self.lsh.insert(doc_id, mh)
        return is_dup

    def run(self, data_file, output_csv):
        output = []
        with tqdm(total=DATA_SIZE, desc="Deduplicating...") as pbar:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if self.skip_text(obj["text"]):
                        continue
                    is_dup = self.deduplicate(obj["text"], obj["id"])
                    output.append([int(is_dup), obj["id"]])
                    pbar.update()
        self.write_results(output, output_csv)
        self.teardown()


def main() -> int:
    args = get_args()
    benchmark_tag = args.input
    benchmark_csv = BENCHMARK_ROOT / DATA_PATH / f"{benchmark_tag}.csv"
    benchmark_jsonl = BENCHMARK_ROOT / DATA_PATH / f"{benchmark_tag}.jsonl"
    if not benchmark_csv.exists() or not benchmark_jsonl.exists():
        raise FileNotFoundError(f"Missing benchmark files for {benchmark_tag}")

    result_dir = BENCHMARK_ROOT / WORK_DIR / benchmark_tag / "char_minhash_lsh_results"
    minhash_root = (
        result_dir
        / "minhashes"
        / f"perm_{args.num_perm}_shingle_{args.shingle_size}_max_{args.max_shingles}"
    )
    output_file = result_dir / f"char_minhash_lsh_{args.sim_threshold}_{args.num_perm}_preds.csv"
    result_file = result_dir / f"char_minhash_lsh_{args.sim_threshold}_{args.num_perm}_score.csv"
    result_dir.mkdir(parents=True, exist_ok=True)

    deduper = CharMinHashLSHDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        shingle_size=int(args.shingle_size),
        max_shingles=int(args.max_shingles),
        minhash_root=minhash_root,
        recompute_minhashes=bool(args.force_compute_minhash),
    )
    deduper.run(str(benchmark_jsonl), str(output_file))
    deduper.score(str(output_file), str(benchmark_csv), str(result_file))
    print(f"Wrote predictions to {output_file}")
    print(f"Wrote score to {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
