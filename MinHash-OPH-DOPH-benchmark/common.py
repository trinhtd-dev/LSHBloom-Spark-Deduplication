import csv
import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import psutil
from datasketch import MinHashLSH
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.autonotebook import tqdm


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_DATA_DIR = REPO_ROOT / "LSH-benchmark" / "benchmark_dfs"
RESULT_DIR_NAMES = {
    "minhash": "minhashlsh",
    "oph": "minhashlsh+oph",
    "doph": "minhashlsh+doph",
}


def ngrams(text, n):
    words = text.split()
    if not words:
        return set()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def skip_text(text):
    return (not isinstance(text, str)) or text is None or text.strip() == "" or text == "None"


def score_predictions(name, output_csv, ground_truth_csv, result_csv):
    df_true = pd.read_csv(ground_truth_csv, sep="|")
    df_pred = pd.read_csv(output_csv)
    df_pred = df_pred.rename(columns={"is_duplicate": "predicted_duplicate"})
    df = pd.merge(df_true, df_pred, on="id", how="left")
    dropped = df.predicted_duplicate.isna().sum()
    print(f"Dropping {dropped} rows for missing predictions")
    df = df.dropna(subset=["predicted_duplicate"])

    y_true = df["is_duplicate"]
    y_pred = df["predicted_duplicate"]
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    header = (
        "name",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "acc",
        "bal_acc",
        "tp",
        "fp",
        "fn",
    )
    output = (
        name,
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        roc_auc_score(y_true, y_pred),
        (y_true == y_pred).mean(),
        balanced_accuracy_score(y_true, y_pred),
        tp,
        fp,
        fn,
    )

    with open(result_csv, "w", newline="", encoding="utf8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerow(output)


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def run_lsh_experiment(
    sketch_name,
    sketcher,
    output_root,
    data_dir,
    tag,
    threshold,
    num_perm,
    ngram,
    limit,
    force,
):
    data_dir = Path(data_dir)
    jsonl_path = data_dir / f"{tag}.jsonl"
    csv_path = data_dir / f"{tag}.csv"
    if not jsonl_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path} or {csv_path}")

    result_dir = Path(output_root) / tag / RESULT_DIR_NAMES.get(sketch_name, f"{sketch_name}_results")
    cache_dir = result_dir / "signatures" / str(num_perm)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    limit_tag = f"_limit_{limit}" if limit > 0 else ""
    prefix = f"{sketch_name}_{threshold}_{num_perm}{limit_tag}"
    pred_file = result_dir / f"{prefix}_preds.csv"
    score_file = result_dir / f"{prefix}_score.csv"
    stats_file = result_dir / f"{prefix}_stats.csv"
    index_file = result_dir / f"{prefix}_index.pkl"

    if pred_file.exists() and score_file.exists() and stats_file.exists() and not force:
        print(f"[skip] {tag} {sketch_name}: existing outputs. Use --force to rerun.")
        return

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, storage_config={"type": "dict"})
    process = psutil.Process(os.getpid())

    rows = []
    docs = 0
    duplicates = 0
    empty_bins_total = 0
    docs_with_empty_bins = 0
    max_empty_bins = 0
    peak_rss = process.memory_info().rss
    t_signature = 0.0
    t_query = 0.0
    t_insert = 0.0
    wall_start = time.perf_counter()

    total_hint = limit if limit > 0 else None
    with open(jsonl_path, "r", encoding="utf8") as fin:
        for line in tqdm(fin, total=total_hint, desc=f"{tag} {sketch_name}"):
            obj = json.loads(line)
            doc_id = obj["id"]
            text = obj.get("text", "")
            if skip_text(text):
                continue

            sig_cache = cache_dir / f"{doc_id}{sketcher.cache_ext}"
            sig_start = time.perf_counter()
            if sig_cache.exists() and not force:
                mh = sketcher.load(sig_cache)
            else:
                mh = sketcher.to_minhash(text, doc_id, ngram)
                sketcher.save(sig_cache, mh)
            empty_bins = int(getattr(sketcher, "last_empty_bins", 0))
            empty_bins_total += empty_bins
            docs_with_empty_bins += int(empty_bins > 0)
            max_empty_bins = max(max_empty_bins, empty_bins)
            t_signature += time.perf_counter() - sig_start

            q_start = time.perf_counter()
            hits = lsh.query(mh)
            t_query += time.perf_counter() - q_start

            is_dup = len(hits) > 0
            if not is_dup:
                ins_start = time.perf_counter()
                lsh.insert(doc_id, mh)
                t_insert += time.perf_counter() - ins_start
            else:
                duplicates += 1

            rows.append([int(is_dup), doc_id])
            docs += 1
            if docs % 250 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
            if limit > 0 and docs >= limit:
                break

    peak_rss = max(peak_rss, process.memory_info().rss)
    wall = time.perf_counter() - wall_start

    with open(pred_file, "w", newline="", encoding="utf8") as fout:
        writer = csv.writer(fout)
        writer.writerow(("is_duplicate", "id"))
        writer.writerows(rows)

    with open(index_file, "wb") as fout:
        pickle.dump(lsh, fout, protocol=pickle.HIGHEST_PROTOCOL)

    score_predictions(sketch_name, pred_file, csv_path, score_file)

    with open(stats_file, "w", newline="", encoding="utf8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            (
                "dataset",
                "sketch",
                "threshold",
                "num_perm",
                "ngram",
                "docs",
                "predicted_duplicates",
                "avg_empty_bins",
                "docs_with_empty_bins",
                "max_empty_bins",
                "wall_sec",
                "signature_sec",
                "query_sec",
                "insert_sec",
                "peak_rss_gb",
                "signature_cache_gb",
                "index_pickle_gb",
            )
        )
        writer.writerow(
            (
                tag,
                sketch_name,
                threshold,
                num_perm,
                ngram,
                docs,
                duplicates,
                empty_bins_total / docs if docs else 0.0,
                docs_with_empty_bins,
                max_empty_bins,
                wall,
                t_signature,
                t_query,
                t_insert,
                peak_rss / (1024**3),
                dir_size(cache_dir) / (1024**3),
                index_file.stat().st_size / (1024**3),
            )
        )

    print(f"[done] {tag} {sketch_name}: score={score_file} stats={stats_file}")
