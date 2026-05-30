from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from tqdm.autonotebook import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
BENCHMARK_ROOT = REPO_ROOT / "LSH-benchmark"

sys.path.insert(0, str(BENCHMARK_ROOT / "dedup"))
sys.path.insert(0, str(BENCHMARK_ROOT / "synthetic_benchmark"))
sys.path.insert(0, str(CURRENT_DIR))

from config import DATA_PATH, DATA_SIZE, WORK_DIR  # type: ignore  # noqa: E402
from dedup_parsing_harness import DedupHarness  # type: ignore  # noqa: E402
from minhash_lsh_cpp import SIGNATURE_INPUT_POLICY, fast_minhash_module, scoped_text  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the C++ rolling-hash MinHash LSH on LSH-benchmark/dedup data."
    )
    parser.add_argument("--input", required=True, help="Benchmark tag, e.g. test_p_0.2")
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=0, help="Override auto-selected LSH bands.")
    parser.add_argument("--rows", type=int, default=0, help="Override auto-selected rows per band.")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingles", type=int, default=150)
    parser.add_argument(
        "--verify-signature-threshold",
        action="store_true",
        help="Require estimated signature similarity >= sim-threshold after an LSH bucket hit.",
    )
    parser.add_argument(
        "--force-compute-minhash",
        action="store_true",
        help="Force recomputing cached rolling signatures.",
    )
    return parser.parse_args()


def _false_positive_probability(threshold: float, bands: int, rows: int) -> float:
    return _integrate_lsh_probability(0.0, threshold, bands, rows)


def _false_negative_probability(threshold: float, bands: int, rows: int) -> float:
    return (1.0 - threshold) - _integrate_lsh_probability(threshold, 1.0, bands, rows)


def _integrate_lsh_probability(start: float, stop: float, bands: int, rows: int) -> float:
    if stop <= start:
        return 0.0
    steps = 512
    width = (stop - start) / steps
    total = 0.0
    for i in range(steps):
        s = start + (i + 0.5) * width
        total += 1.0 - (1.0 - s**rows) ** bands
    return total * width


def optimal_lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("--sim-threshold must be in (0, 1)")
    if num_perm <= 0:
        raise ValueError("--num-perm must be positive")

    min_error = float("inf")
    opt = (1, num_perm)
    for bands in range(1, num_perm + 1):
        max_rows = num_perm // bands
        for rows in range(1, max_rows + 1):
            fp = _false_positive_probability(threshold, bands, rows)
            fn = _false_negative_probability(threshold, bands, rows)
            error = 0.5 * fp + 0.5 * fn
            if error < min_error:
                min_error = error
                opt = (bands, rows)
    return opt


def band_keys(signature: tuple[int, ...], bands: int, rows: int) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for band in range(bands):
        start = band * rows
        chunk = signature[start : start + rows]
        if len(chunk) != rows:
            continue
        digest = hashlib.blake2b(
            ",".join(str(v) for v in chunk).encode("ascii"),
            digest_size=12,
        ).hexdigest()
        out.append((band, digest))
    return out


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...], used: int) -> float:
    if used <= 0:
        return 0.0
    return sum(1 for i in range(used) if left[i] == right[i]) / used


class FastRollingMinHashLSHDeduper(DedupHarness):
    def __init__(
        self,
        sim_threshold: float,
        num_perm: int,
        bands: int,
        rows: int,
        shingle_size: int,
        max_shingles: int,
        signature_root: Path,
        recompute_signatures: bool = False,
        verify_signature_threshold: bool = False,
    ):
        super().__init__("fast_minhash_lsh")
        self.T = float(sim_threshold)
        self.num_perm = int(num_perm)
        self.bands = int(bands)
        self.rows = int(rows)
        self.shingle_size = int(shingle_size)
        self.max_shingles = int(max_shingles)
        self.signature_root = signature_root
        self.recompute_signatures = bool(recompute_signatures)
        self.verify_signature_threshold = bool(verify_signature_threshold)
        self.used_perm = self.bands * self.rows

        if self.used_perm > self.num_perm:
            raise ValueError("--bands * --rows cannot exceed --num-perm")
        if self.bands <= 0 or self.rows <= 0:
            raise ValueError("LSH bands and rows must be positive")

        self.fast_minhash = fast_minhash_module()
        self.tables: list[defaultdict[str, set[str]]] = [defaultdict(set) for _ in range(self.bands)]
        self.signatures: dict[str, tuple[int, ...]] = {}
        self.signature_root.mkdir(parents=True, exist_ok=True)

    def get_signature(self, text: str, doc_id: str) -> tuple[int, ...]:
        cache_key = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=16).hexdigest()
        cache_file = self.signature_root / f"{cache_key}.pkl"
        if not self.recompute_signatures and cache_file.exists():
            with cache_file.open("rb") as f:
                signature = pickle.load(f)
            return tuple(int(v) for v in signature)

        scoped = scoped_text(text, self.shingle_size, self.max_shingles)
        if len(scoped) < self.shingle_size:
            signature = ()
        else:
            signature = tuple(
                int(v)
                for v in self.fast_minhash.rolling_char_signature(
                    scoped,
                    self.shingle_size,
                    self.num_perm,
                )
            )

        with cache_file.open("wb") as f:
            pickle.dump(signature, f)
        return signature

    def query(self, signature: tuple[int, ...]) -> set[str]:
        candidate_ids: set[str] = set()
        for band, digest in band_keys(signature, self.bands, self.rows):
            candidate_ids.update(self.tables[band].get(digest, set()))
        return candidate_ids

    def insert(self, doc_id: str, signature: tuple[int, ...]) -> None:
        for band, digest in band_keys(signature, self.bands, self.rows):
            self.tables[band][digest].add(doc_id)
        self.signatures[doc_id] = signature

    def deduplicate(self, text: str, id: int) -> bool:
        doc_id = str(id)
        signature = self.get_signature(text, doc_id)
        if len(signature) < self.used_perm:
            return False

        candidates = self.query(signature)
        if self.verify_signature_threshold:
            is_dup = any(
                signature_similarity(signature, self.signatures[candidate_id], self.used_perm) >= self.T
                for candidate_id in candidates
            )
        else:
            is_dup = bool(candidates)

        if not is_dup:
            self.insert(doc_id, signature)
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


def write_metadata(path: Path, rows: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows.items())


def main() -> int:
    args = get_args()
    if args.bands or args.rows:
        if not args.bands or not args.rows:
            raise ValueError("--bands and --rows must be provided together")
        bands, rows = int(args.bands), int(args.rows)
    else:
        bands, rows = optimal_lsh_params(float(args.sim_threshold), int(args.num_perm))

    benchmark_tag = args.input
    benchmark_csv = BENCHMARK_ROOT / DATA_PATH / f"{benchmark_tag}.csv"
    benchmark_jsonl = BENCHMARK_ROOT / DATA_PATH / f"{benchmark_tag}.jsonl"
    if not benchmark_csv.exists() or not benchmark_jsonl.exists():
        raise FileNotFoundError(
            f"Missing benchmark files: {benchmark_csv} and/or {benchmark_jsonl}"
        )

    result_dir = BENCHMARK_ROOT / WORK_DIR / benchmark_tag / "fast_minhash_lsh_results"
    signature_root = (
        result_dir
        / "signatures"
        / f"perm_{args.num_perm}_shingle_{args.shingle_size}_max_{args.max_shingles}"
    )
    output_file = result_dir / f"fast_minhash_lsh_{args.sim_threshold}_{args.num_perm}_preds.csv"
    result_file = result_dir / f"fast_minhash_lsh_{args.sim_threshold}_{args.num_perm}_score.csv"
    metadata_file = result_dir / f"fast_minhash_lsh_{args.sim_threshold}_{args.num_perm}_meta.csv"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[fast_minhash_lsh]",
        f"threshold={float(args.sim_threshold):.3f}",
        f"num_perm={args.num_perm}",
        f"bands={bands}",
        f"rows={rows}",
        f"used_perm={bands * rows}",
        f"shingle_size={args.shingle_size}",
        f"max_shingles={args.max_shingles}",
        f"verify_signature_threshold={args.verify_signature_threshold}",
    )

    deduper = FastRollingMinHashLSHDeduper(
        sim_threshold=float(args.sim_threshold),
        num_perm=int(args.num_perm),
        bands=bands,
        rows=rows,
        shingle_size=int(args.shingle_size),
        max_shingles=int(args.max_shingles),
        signature_root=signature_root,
        recompute_signatures=bool(args.force_compute_minhash),
        verify_signature_threshold=bool(args.verify_signature_threshold),
    )
    deduper.run(str(benchmark_jsonl), str(output_file))
    deduper.score(str(output_file), str(benchmark_csv), str(result_file))
    write_metadata(
        metadata_file,
        {
            "name": "fast_minhash_lsh",
            "signature_input_policy": SIGNATURE_INPUT_POLICY,
            "sim_threshold": args.sim_threshold,
            "num_perm": args.num_perm,
            "bands": bands,
            "rows": rows,
            "used_perm": bands * rows,
            "shingle_size": args.shingle_size,
            "max_shingles": args.max_shingles,
            "verify_signature_threshold": args.verify_signature_threshold,
            "output_file": output_file,
            "result_file": result_file,
        },
    )
    print(f"Wrote predictions to {output_file}")
    print(f"Wrote score to {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
