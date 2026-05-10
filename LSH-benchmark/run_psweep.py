from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# Adjust if you only want one algorithm.
ALGORITHMS = [
    (
        "lsh",
        ["dedup/lsh/lsh.py", "--sim-threshold", "0.8", "--num-perm", "128", "--ngram", "1"],
        "lsh_results",
        "lsh_0.8_128_score.csv",
    ),
    (
        "lsh_bloom",
        ["dedup/lsh/lsh_bloom.py", "--sim-threshold", "0.8", "--num-perm", "128", "--ngram", "1"],
        "lsh_bloom_results",
        "lsh_bloom_0.8_128_score.csv",
    ),
]

OUTPUT_SUMMARY = BASE_DIR / "psweep_summary.csv"


def dataset_tag(p: float) -> str:
    return f"my_benchmark_p{int(round(p * 10)):02d}"


def score_file(dataset: str, results_dir: str, filename: str) -> Path:
    return BASE_DIR / dataset / results_dir / filename


def run_cmd(cmd: list[str]) -> int:
    print("\n==>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


def read_score(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main() -> int:
    rows = []
    for p in [i / 10 for i in range(1, 10)]:
        tag = dataset_tag(p)
        jsonl_path = BASE_DIR / "benchmark_dfs" / f"{tag}.jsonl"
        csv_path = BASE_DIR / "benchmark_dfs" / f"{tag}.csv"

        if not jsonl_path.exists() or not csv_path.exists():
            print(f"[skip] missing benchmark files for {tag}")
            continue

        for algo_name, script_args, results_dir, score_name in ALGORITHMS:
            cmd = [PYTHON, *script_args, "--input", tag]
            rc = run_cmd(cmd)
            if rc != 0:
                print(f"[error] {algo_name} failed on {tag}")
                continue

            sfile = score_file(tag, results_dir, score_name)
            score = read_score(sfile)
            if not score:
                print(f"[warn] score file missing for {algo_name} on {tag}: {sfile}")
                continue

            rows.append(
                {
                    "dataset": tag,
                    "p": p,
                    "algorithm": algo_name,
                    "precision": score.get("precision", ""),
                    "recall": score.get("recall", ""),
                    "f1": score.get("f1", ""),
                    "auc_roc": score.get("auc_roc", ""),
                    "acc": score.get("acc", ""),
                    "bal_acc": score.get("bal_acc", ""),
                    "tp": score.get("tp", ""),
                    "fp": score.get("fp", ""),
                    "fn": score.get("fn", ""),
                }
            )

    if rows:
        with OUTPUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote summary to {OUTPUT_SUMMARY}")
    else:
        print("\nNo results collected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
