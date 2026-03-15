
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
MINHASH_SEED = 42


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    return TOKEN_RE.findall(clean_text(text).lower())


def make_shingles_from_words(words: Sequence[str], n: int) -> List[str]:
    if not words:
        return [""]
    if n <= 1:
        # unique words to reduce updates
        return list(dict.fromkeys(words))
    if len(words) < n:
        return [" ".join(words)]
    out: List[str] = []
    seen = set()
    for i in range(len(words) - n + 1):
        sh = " ".join(words[i : i + n])
        if sh not in seen:
            out.append(sh)
            seen.add(sh)
    return out


def validate_doc_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["doc_id", "source_doc_id", "text", "variant_family"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    out = df.copy()
    out["doc_id"] = out["doc_id"].astype(str)
    out["source_doc_id"] = out["source_doc_id"].astype(str)
    out["text"] = out["text"].fillna("").astype(str).map(clean_text)
    out["variant_family"] = out["variant_family"].fillna("unknown").astype(str)

    if "n_words" not in out.columns:
        out["n_words"] = out["text"].map(lambda x: len(tokenize_words(x)))

    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def build_stream_eval_dataset(
    doc_df: pd.DataFrame,
    prevalence: float,
    total_docs: int,
    seed: int,
) -> pd.DataFrame:
    """
    Stream/document-level eval:
    - unique docs = originals from selected sources
    - duplicates = non-original variants from those same sources
    - ground truth duplicate at time t: source_doc_id has appeared earlier in stream
    """
    rng = random.Random(seed)

    source_to_group: Dict[str, Dict[str, object]] = {}
    for sid, group in doc_df.groupby("source_doc_id"):
        g = group.copy()
        originals = g[g["variant_family"] == "original"]
        dups = g[g["variant_family"] != "original"]

        if len(originals) == 0 or len(dups) == 0:
            continue

        source_to_group[str(sid)] = {
            "original": originals.iloc[0].to_dict(),
            "duplicates": dups.to_dict("records"),
        }

    available_sources = list(source_to_group.keys())
    rng.shuffle(available_sources)

    if not available_sources:
        raise ValueError("No valid source groups found. Need at least one original and one duplicate per source.")

    n_unique = max(1, int(round(total_docs * (1.0 - prevalence))))
    n_unique = min(n_unique, len(available_sources))
    n_dups_target = total_docs - n_unique

    selected_sources = available_sources[:n_unique]

    originals = []
    dup_pool: Dict[str, deque] = {}
    for sid in selected_sources:
        originals.append(source_to_group[sid]["original"])
        local_dups = list(source_to_group[sid]["duplicates"])
        rng.shuffle(local_dups)
        dup_pool[sid] = deque(local_dups)

    max_possible_dups = sum(len(v) for v in dup_pool.values())
    if n_dups_target > max_possible_dups:
        n_dups_target = max_possible_dups

    selected_dups = []
    active_sources = selected_sources.copy()
    rng.shuffle(active_sources)

    while len(selected_dups) < n_dups_target and active_sources:
        next_active = []
        for sid in active_sources:
            if len(selected_dups) >= n_dups_target:
                break
            if dup_pool[sid]:
                selected_dups.append(dup_pool[sid].popleft())
            if dup_pool[sid]:
                next_active.append(sid)
        rng.shuffle(next_active)
        active_sources = next_active

    # Interleave originals and duplicates to make the stream less trivial
    stream_rows = []
    remaining_originals = originals.copy()
    remaining_dups = selected_dups.copy()
    rng.shuffle(remaining_originals)
    rng.shuffle(remaining_dups)

    while remaining_originals or remaining_dups:
        can_take_dup = len(remaining_dups) > 0 and len(stream_rows) > 0
        take_dup = False

        if can_take_dup and remaining_originals:
            current_prev = len(remaining_dups) / max(1, len(remaining_originals) + len(remaining_dups))
            take_dup = rng.random() < current_prev
        elif can_take_dup:
            take_dup = True

        if take_dup:
            stream_rows.append(remaining_dups.pop())
        elif remaining_originals:
            stream_rows.append(remaining_originals.pop())
        elif remaining_dups:
            stream_rows.append(remaining_dups.pop())

    stream_df = pd.DataFrame(stream_rows).reset_index(drop=True)
    if len(stream_df) == 0:
        raise ValueError("Built empty stream.")
    return stream_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class SignatureCache:
    def __init__(self, doc_df: pd.DataFrame, doc_ids: Iterable[str], ngram_n: int, num_perm: int):
        self.doc_df = doc_df[doc_df["doc_id"].isin(set(map(str, doc_ids)))][["doc_id", "text"]].copy()
        self.doc_df["doc_id"] = self.doc_df["doc_id"].astype(str)
        self.doc_df = self.doc_df.sort_values("doc_id").reset_index(drop=True)
        self.ngram_n = int(ngram_n)
        self.num_perm = int(num_perm)
        self.obj_cache: Dict[str, MinHash] = {}

    def build_minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm, seed=MINHASH_SEED)
        shingles = make_shingles_from_words(tokenize_words(text), self.ngram_n)
        for sh in shingles:
            mh.update(sh.encode("utf-8", errors="ignore"))
        return mh

    def get_mh(self, doc_id: str) -> MinHash:
        doc_id = str(doc_id)
        if doc_id in self.obj_cache:
            return self.obj_cache[doc_id]
        row = self.doc_df[self.doc_df["doc_id"] == doc_id]
        if row.empty:
            raise KeyError(f"doc_id not found in SignatureCache: {doc_id}")
        text = row.iloc[0]["text"]
        mh = self.build_minhash(text)
        self.obj_cache[doc_id] = mh
        return mh


def run_minhash_lsh_stream_eval(stream_df: pd.DataFrame, sig_cache: SignatureCache, threshold: float) -> Dict[str, float]:
    lsh = MinHashLSH(threshold=float(threshold), num_perm=sig_cache.num_perm)

    seen_source_ids = set()
    y_true: List[int] = []
    y_pred: List[int] = []

    query_sec = 0.0
    insert_sec = 0.0
    total_hits = 0
    docs_with_hits = 0

    for row in stream_df[["doc_id", "source_doc_id"]].itertuples(index=False):
        doc_id = str(row.doc_id)
        sid = str(row.source_doc_id)

        y_true.append(1 if sid in seen_source_ids else 0)
        mh = sig_cache.get_mh(doc_id)

        t0 = time.perf_counter()
        hits = lsh.query(mh)
        query_sec += time.perf_counter() - t0

        pred = 1 if len(hits) > 0 else 0
        y_pred.append(pred)

        if hits:
            docs_with_hits += 1
            total_hits += len(hits)

        t0 = time.perf_counter()
        lsh.insert(doc_id, mh)
        insert_sec += time.perf_counter() - t0

        seen_source_ids.add(sid)

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics.update(
        {
            "n_docs": int(len(stream_df)),
            "n_duplicates_true": int(y_true_arr.sum()),
            "n_predicted_duplicates": int(y_pred_arr.sum()),
            "docs_with_hits": int(docs_with_hits),
            "total_hit_count": int(total_hits),
            "query_sec": float(query_sec),
            "insert_sec": float(insert_sec),
        }
    )
    return metrics


def evaluate_prevalence_grid_from_streams(
    streams: Dict[float, pd.DataFrame],
    sig_cache: SignatureCache,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for prevalence, stream_df in streams.items():
        metrics = run_minhash_lsh_stream_eval(stream_df=stream_df, sig_cache=sig_cache, threshold=threshold)
        metrics["prevalence"] = float(prevalence)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("prevalence").reset_index(drop=True)


def build_streams_for_grid(
    doc_df: pd.DataFrame,
    prevalence_grid: Sequence[float],
    total_docs: int,
    seed_base: int,
) -> Tuple[Dict[float, pd.DataFrame], List[str]]:
    streams: Dict[float, pd.DataFrame] = {}
    all_doc_ids = set()

    for i, prevalence in enumerate(prevalence_grid):
        stream_df = build_stream_eval_dataset(
            doc_df=doc_df,
            prevalence=float(prevalence),
            total_docs=int(total_docs),
            seed=int(seed_base + i),
        )
        streams[float(prevalence)] = stream_df
        all_doc_ids.update(stream_df["doc_id"].astype(str).tolist())

    return streams, sorted(all_doc_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MinHashLSH on parquet benchmark files.")
    parser.add_argument("--tuning-parquet", default="data/tuning_docs.parquet")
    parser.add_argument("--test-parquet", default="data/test_docs.parquet")
    parser.add_argument("--output-dir", default="results/minhashlsh_stream_eval")

    parser.add_argument("--prevalence-grid", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--num-perm-grid", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--ngram-grid", nargs="+", type=int, default=[3])

    parser.add_argument("--tuning-stream-docs", type=int, default=4000)
    parser.add_argument("--test-stream-docs", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tuning_path = Path(args.tuning_parquet)
    test_path = Path(args.test_parquet)

    if not tuning_path.exists():
        raise FileNotFoundError(f"Missing tuning parquet: {tuning_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test parquet: {test_path}")

    print(f"[INFO] Loading tuning parquet: {tuning_path}")
    tuning_docs = validate_doc_df(pd.read_parquet(tuning_path), "tuning_parquet")
    print(f"[INFO] Loading test parquet:   {test_path}")
    test_docs = validate_doc_df(pd.read_parquet(test_path), "test_parquet")

    print(f"[INFO] tuning_docs: {len(tuning_docs):,} rows | sources: {tuning_docs['source_doc_id'].nunique():,}")
    print(f"[INFO] test_docs:   {len(test_docs):,} rows | sources: {test_docs['source_doc_id'].nunique():,}")

    tuning_streams, tuning_stream_doc_ids = build_streams_for_grid(
        doc_df=tuning_docs,
        prevalence_grid=args.prevalence_grid,
        total_docs=args.tuning_stream_docs,
        seed_base=args.seed + 5000,
    )
    test_streams, test_stream_doc_ids = build_streams_for_grid(
        doc_df=test_docs,
        prevalence_grid=args.prevalence_grid,
        total_docs=args.test_stream_docs,
        seed_base=args.seed + 9000,
    )

    print(f"[INFO] tuning streams: {len(tuning_streams)} | unique docs used: {len(tuning_stream_doc_ids):,}")
    print(f"[INFO] test streams:   {len(test_streams)} | unique docs used: {len(test_stream_doc_ids):,}")

    tuning_rows = []
    tuning_sig_caches: Dict[Tuple[int, int], SignatureCache] = {}

    for ngram_n in args.ngram_grid:
        for num_perm in args.num_perm_grid:
            cache_key = (int(ngram_n), int(num_perm))
            sig_cache = SignatureCache(
                doc_df=tuning_docs,
                doc_ids=tuning_stream_doc_ids,
                ngram_n=int(ngram_n),
                num_perm=int(num_perm),
            )
            tuning_sig_caches[cache_key] = sig_cache

            for threshold in args.threshold_grid:
                prev_df = evaluate_prevalence_grid_from_streams(
                    streams=tuning_streams,
                    sig_cache=sig_cache,
                    threshold=float(threshold),
                )
                summary = {
                    "ngram_n": int(ngram_n),
                    "num_perm": int(num_perm),
                    "threshold": float(threshold),
                    "mean_precision": float(prev_df["precision"].mean()),
                    "mean_recall": float(prev_df["recall"].mean()),
                    "mean_f1": float(prev_df["f1"].mean()),
                    "min_f1": float(prev_df["f1"].min()),
                    "max_f1": float(prev_df["f1"].max()),
                    "mean_query_sec": float(prev_df["query_sec"].mean()),
                    "mean_insert_sec": float(prev_df["insert_sec"].mean()),
                }
                tuning_rows.append(summary)
                print(
                    "[TUNE]",
                    f"ngram={ngram_n}",
                    f"perm={num_perm}",
                    f"thr={threshold}",
                    f"mean_f1={summary['mean_f1']:.4f}",
                    f"mean_p={summary['mean_precision']:.4f}",
                    f"mean_r={summary['mean_recall']:.4f}",
                )

    stream_tuning_df = (
        pd.DataFrame(tuning_rows)
        .sort_values(["mean_f1", "mean_precision", "mean_recall"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    stream_tuning_df.to_csv(output_dir / "stream_tuning_results.csv", index=False)

    best_stream_cfg = stream_tuning_df.iloc[0].to_dict()
    best_ngram_n = int(best_stream_cfg["ngram_n"])
    best_num_perm = int(best_stream_cfg["num_perm"])
    best_threshold = float(best_stream_cfg["threshold"])

    print("\n[INFO] Best tuning config:")
    print(json.dumps(best_stream_cfg, ensure_ascii=False, indent=2))

    test_sig_cache = SignatureCache(
        doc_df=test_docs,
        doc_ids=test_stream_doc_ids,
        ngram_n=best_ngram_n,
        num_perm=best_num_perm,
    )
    stream_test_df = evaluate_prevalence_grid_from_streams(
        streams=test_streams,
        sig_cache=test_sig_cache,
        threshold=best_threshold,
    )
    stream_test_df.to_csv(output_dir / "stream_test_results.csv", index=False)

    test_mean = stream_test_df[["precision", "recall", "f1"]].mean().to_dict()

    with open(output_dir / "best_configs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "prevalence_grid": list(map(float, args.prevalence_grid)),
                "threshold_grid": list(map(float, args.threshold_grid)),
                "num_perm_grid": list(map(int, args.num_perm_grid)),
                "ngram_grid": list(map(int, args.ngram_grid)),
                "best_stream_cfg": best_stream_cfg,
                "stream_test_mean_metrics": test_mean,
                "tuning_parquet": str(tuning_path),
                "test_parquet": str(test_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[INFO] Saved:")
    print(f"  - {output_dir / 'stream_tuning_results.csv'}")
    print(f"  - {output_dir / 'stream_test_results.csv'}")
    print(f"  - {output_dir / 'best_configs.json'}")

    print("\n[INFO] Stream test mean metrics:")
    print(json.dumps(test_mean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
