"""
Script so sánh F1/Precision/Recall của các thuật toán LSH Bloom variants:
  - lsh_bloom        (Standard Bloom Filter)
  - lsh_blocked_bloom (Blocked Bloom Filter - cache-line optimized)
  - lsh_blowchoc     (BlowChoc Filter - single word-level hashing)

Chạy trên synthetic benchmark dataset có sẵn ground truth.
Kết quả in ra bảng so sánh và lưu vào CSV.
"""

import os
import sys
import csv
import time
import json
import pickle
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LSH_DIR = os.path.join(CURRENT_DIR, "dedup", "lsh")
SYNTH_DIR = os.path.join(CURRENT_DIR, "synthetic_benchmark")
DEDUP_DIR = os.path.join(CURRENT_DIR, "dedup")

sys.path.insert(0, LSH_DIR)
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, DEDUP_DIR)

import pandas as pd
from pathlib import Path
from config import DATA_PATH, WORK_DIR, DATA_SIZE
from datasketch import MinHash


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_minhash(text: str, num_perm: int, ngram: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    words = text.split()
    if len(words) < ngram:
        shingles = set(words)
    else:
        shingles = set(" ".join(words[i:i+ngram]) for i in range(len(words) - ngram + 1))
    if not shingles:
        shingles = {"__EMPTY__"}
    for s in shingles:
        mh.update(s.encode("utf8"))
    return mh


def _optimal_param(threshold, num_perm, fp_w=0.5, fn_w=0.5):
    from scipy.integrate import quad as integrate
    def fpp(threshold, b, r):
        p = lambda s: 1 - (1 - s**float(r))**float(b)
        return integrate(p, 0.0, threshold)[0]
    def fnp(threshold, b, r):
        p = lambda s: 1 - (1 - (1 - s**float(r))**float(b))
        return integrate(p, threshold, 1.0)[0]
    min_err = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        for r in range(1, int(num_perm / b) + 1):
            err = fpp(threshold, b, r) * fp_w + fnp(threshold, b, r) * fn_w
            if err < min_err:
                min_err = err
                opt = (b, r)
    return opt


def score_results(name, preds_csv, gt_csv):
    from sklearn.metrics import precision_score, recall_score, f1_score
    df_gt  = pd.read_csv(gt_csv, sep="|")
    df_pred = pd.read_csv(preds_csv)
    df_pred = df_pred.rename(columns={"is_duplicate": "predicted_duplicate"})
    df = pd.merge(df_gt, df_pred, on="id", how="left").dropna(subset=["predicted_duplicate"])
    y_true = df["is_duplicate"].astype(int)
    y_pred = df["predicted_duplicate"].astype(int)
    return {
        "name":      name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tp":        int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp":        int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn":        int(((y_true == 1) & (y_pred == 0)).sum()),
        "n_docs":    len(df),
    }


# ─── Per-algo runners ─────────────────────────────────────────────────────────

def run_standard_bloom(data_jsonl, gt_csv, result_dir, threshold, num_perm, ngram, fp_rate):
    """Standard Bloom Filter (original lsh_bloom)."""
    from lsh_bloom import LSHBloomDeduper
    minhash_root = os.path.join(result_dir, "standard_bloom", f"minhashes_{num_perm}")
    save_dir = os.path.join(result_dir, "standard_bloom", f"filter_{threshold}_{num_perm}")
    preds_csv = os.path.join(result_dir, "standard_bloom", f"preds_{threshold}_{num_perm}.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    deduper = LSHBloomDeduper(
        n=DATA_SIZE, sim_threshold=threshold, num_perm=num_perm,
        minhash_root=minhash_root, save_dir=save_dir,
        recompute_minhashes=True, fp=fp_rate, ngram=ngram,
    )
    t0 = time.perf_counter()
    deduper.run(data_jsonl, preds_csv)
    elapsed = time.perf_counter() - t0

    metrics = score_results("lsh_bloom", preds_csv, gt_csv)
    metrics["wall_sec"] = round(elapsed, 2)
    return metrics


def run_blocked_bloom(data_jsonl, gt_csv, result_dir, threshold, num_perm, ngram, fp_rate):
    """Blocked Bloom Filter."""
    from lsh_blocked_bloom import LSHBlockedBloomDeduper
    minhash_root = os.path.join(result_dir, "blocked_bloom", f"minhashes_{num_perm}")
    save_dir = os.path.join(result_dir, "blocked_bloom", f"filter_{threshold}_{num_perm}")
    preds_csv = os.path.join(result_dir, "blocked_bloom", f"preds_{threshold}_{num_perm}.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    deduper = LSHBlockedBloomDeduper(
        n=DATA_SIZE, sim_threshold=threshold, num_perm=num_perm,
        minhash_root=minhash_root, save_dir=save_dir,
        recompute_minhashes=True, fp=fp_rate, ngram=ngram,
    )
    t0 = time.perf_counter()
    deduper.run(data_jsonl, preds_csv)
    elapsed = time.perf_counter() - t0

    metrics = score_results("lsh_blocked_bloom", preds_csv, gt_csv)
    metrics["wall_sec"] = round(elapsed, 2)
    return metrics


def run_blowchoc(data_jsonl, gt_csv, result_dir, threshold, num_perm, ngram, fp_rate):
    """BlowChoc Filter."""
    from lsh_blowchoc import LSHBlowChocDeduper
    minhash_root = os.path.join(result_dir, "blowchoc", f"minhashes_{num_perm}")
    save_dir = os.path.join(result_dir, "blowchoc", f"filter_{threshold}_{num_perm}")
    preds_csv = os.path.join(result_dir, "blowchoc", f"preds_{threshold}_{num_perm}.csv")
    os.makedirs(minhash_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    deduper = LSHBlowChocDeduper(
        n=DATA_SIZE, sim_threshold=threshold, num_perm=num_perm,
        minhash_root=minhash_root, save_dir=save_dir,
        recompute_minhashes=True, fp=fp_rate, ngram=ngram,
    )
    t0 = time.perf_counter()
    deduper.run(data_jsonl, preds_csv)
    elapsed = time.perf_counter() - t0

    metrics = score_results("lsh_blowchoc", preds_csv, gt_csv)
    metrics["wall_sec"] = round(elapsed, 2)
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_table(rows: list):
    if not rows:
        return
    headers = ["algo", "threshold", "num_perm", "f1", "precision", "recall", "tp", "fp", "fn", "wall_sec"]
    col_w = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    sep = "+" + "+".join("-" * (col_w[h] + 2) for h in headers) + "+"
    header_row = "|" + "|".join(f" {h:<{col_w[h]}} " for h in headers) + "|"
    print(sep)
    print(header_row)
    print(sep)
    for r in rows:
        print("|" + "|".join(f" {str(r.get(h, '')):<{col_w[h]}} " for h in headers) + "|")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Compare F1 of lsh_bloom / lsh_blocked_bloom / lsh_blowchoc")
    parser.add_argument("--input", default="test_p_0.5",
                        help="Benchmark tag (e.g. test_p_0.5). Must match files in benchmark_dfs/")
    parser.add_argument("--thresholds", default="0.5",
                        help="Comma-separated Jaccard thresholds (e.g. 0.3,0.5,0.7)")
    parser.add_argument("--num-perm", type=int, default=128,
                        help="Number of MinHash permutations")
    parser.add_argument("--ngram", type=int, default=5,
                        help="N-gram size for shingling")
    parser.add_argument("--fp-rate", type=float, default=1e-5,
                        help="Target false positive rate for Bloom filters")
    parser.add_argument("--algos", default="lsh_bloom,lsh_blocked_bloom,lsh_blowchoc",
                        help="Comma-separated list of algos to run")
    parser.add_argument("--out-dir", default="quality_results",
                        help="Directory to save prediction CSVs and summary")
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    data_jsonl = os.path.join(DATA_PATH, f"{args.input}.jsonl")
    gt_csv = os.path.join(DATA_PATH, f"{args.input}.csv")

    if not os.path.exists(data_jsonl):
        print(f"[ERROR] Data file not found: {data_jsonl}")
        print(f"  Make sure '{args.input}.jsonl' exists in '{DATA_PATH}/'")
        return 1
    if not os.path.exists(gt_csv):
        print(f"[ERROR] Ground truth file not found: {gt_csv}")
        return 1

    result_dir = os.path.join(args.out_dir, args.input)
    os.makedirs(result_dir, exist_ok=True)

    all_rows = []
    runner_map = {
        "lsh_bloom": run_standard_bloom,
        "lsh_blocked_bloom": run_blocked_bloom,
        "lsh_blowchoc": run_blowchoc,
    }

    for threshold in thresholds:
        for algo in algos:
            if algo not in runner_map:
                print(f"[WARN] Unknown algo: {algo}, skipping.")
                continue
            print(f"\n{'='*60}")
            print(f"  Running: {algo}  threshold={threshold}  num_perm={args.num_perm}")
            print(f"{'='*60}")
            try:
                metrics = runner_map[algo](
                    data_jsonl, gt_csv, result_dir,
                    threshold=threshold,
                    num_perm=args.num_perm,
                    ngram=args.ngram,
                    fp_rate=args.fp_rate,
                )
                row = {
                    "algo": algo,
                    "threshold": threshold,
                    "num_perm": args.num_perm,
                    **metrics,
                }
                all_rows.append(row)
                print(f"  → F1={metrics['f1']}  Precision={metrics['precision']}  Recall={metrics['recall']}  Time={metrics['wall_sec']}s")
            except Exception as e:
                print(f"  [ERROR] {algo} failed: {e}")
                import traceback; traceback.print_exc()

    if not all_rows:
        print("[ERROR] No results collected.")
        return 1

    # Print table
    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}")
    print_table(all_rows)

    # Save summary CSV
    summary_path = os.path.join(result_dir, "comparison_summary.csv")
    fieldnames = ["algo", "threshold", "num_perm", "f1", "precision", "recall", "tp", "fp", "fn", "wall_sec", "n_docs"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[✓] Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
