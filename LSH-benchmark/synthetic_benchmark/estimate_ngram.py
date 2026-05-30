"""
Script that gets a probabilistic estimate of the number of n-grams in a corpus.
Useful for providing this information to deduplication routines that need to set
the size of a Bloom Filter, for instance.
"""

<<<<<<< HEAD
=======
import argparse
>>>>>>> 8db5542e13743bc15b6e4dbce8cba9484e46e6b6
import os
import sys
import json
from glob import glob
import re
from typing import List
import math
import pandas as pd
from collections import defaultdict
from config import *
from tqdm.autonotebook import tqdm

def tokenize(s: str) -> List[str]:
    # Split the string into word boundaries
    words = re.findall(r'\b\w+\b', s)
    
    # Filter out whitespace-only tokens
    return list(filter(lambda w: not all(c.isspace() for c in w), words))

def ngram(tokens: List[str], size: int, stride: int) -> List[str]:
    """
    Constructs a list of ngrams from a list of  tokens
    """
    ngrams = []
    for i in range(0, len(tokens) - size + 1, size*stride):
        ngram = ' '.join(tokens[i:i+size])
        ngrams.append(ngram)
    return ngrams

def count_ngram(doc: str, ngram_size: int) -> int:
    return len(ngram(tokenize(doc), ngram_size, stride=1))

<<<<<<< HEAD
if __name__ == "__main__":
    N = 1000
    SIZES = [1, 2, 5, 7, 13, 26]
    counts = defaultdict(lambda: 0)

    benchmark_csv_files = glob(f"{DATA_PATH}/{DATA_TAG}_bench*.csv")

    for benchmark_csv in benchmark_csv_files:
        benchmark_tag = os.path.basename(benchmark_csv).split('.csv')[0]
        df = pd.read_csv(benchmark_csv, sep="|").sample(n=N)
        df = df[["id", "new_text"]]
        num_iter = N * len(SIZES)
        with tqdm(total=num_iter, desc=f"({benchmark_tag}) Estimating ngram counts...") as pbar:
            for row in df.itertuples(index=False):
                for sz in SIZES:
                    num_ngrams = count_ngram(row.new_text, sz)
                    counts[sz] += num_ngrams
                    pbar.update()
        
            # compute estimates
            for sz in SIZES:
                mean = math.ceil(counts[sz] / N)
=======

def _extract_text(record: dict) -> str | None:
    for key in ("text", "content", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    pages = record.get("pages")
    if isinstance(pages, list):
        parts = []
        for page in pages:
            if isinstance(page, dict):
                text = page.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Benchmark tag (e.g., test_p_0.5).")
    parser.add_argument("--ngram-size", type=int, default=0, help="If set, only estimate this n-gram size.")
    parser.add_argument("--sample", type=int, default=1000, help="Number of rows to sample.")
    args = parser.parse_args()

    N = int(args.sample)
    SIZES = [1, 2, 5, 7, 13, 26]
    if args.ngram_size:
        SIZES = [int(args.ngram_size)]

    if args.input:
        benchmark_csv_files = [os.path.join(DATA_PATH, f"{args.input}.csv")]
    else:
        benchmark_csv_files = glob(f"{DATA_PATH}/{DATA_TAG}_bench*.csv")

    for benchmark_csv in benchmark_csv_files:
        if not os.path.exists(benchmark_csv):
            print(f"[skip] missing benchmark CSV: {benchmark_csv}")
            continue
        counts = defaultdict(lambda: 0)
        benchmark_tag = os.path.basename(benchmark_csv).split('.csv')[0]
        df = pd.read_csv(benchmark_csv, sep="|")
        if len(df) > N:
            df = df.sample(n=N)
        text_col = "new_text" if "new_text" in df.columns else "text"
        if text_col in df.columns:
            df = df[["id", text_col]]
            df = df.rename(columns={text_col: "new_text"})
            num_iter = len(df) * len(SIZES)
            with tqdm(total=num_iter, desc=f"({benchmark_tag}) Estimating ngram counts...") as pbar:
                for row in df.itertuples(index=False):
                    for sz in SIZES:
                        num_ngrams = count_ngram(row.new_text, sz)
                        counts[sz] += num_ngrams
                        pbar.update()
        else:
            jsonl_path = os.path.join(DATA_PATH, f"{benchmark_tag}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"[skip] missing text column and JSONL: {benchmark_csv}")
                continue
            sampled = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _extract_text(record)
                    if text is None:
                        continue
                    sampled.append(text)
                    if len(sampled) >= N:
                        break
            if not sampled:
                print(f"[skip] no text found in JSONL: {jsonl_path}")
                continue
            num_iter = len(sampled) * len(SIZES)
            with tqdm(total=num_iter, desc=f"({benchmark_tag}) Estimating ngram counts...") as pbar:
                for text in sampled:
                    for sz in SIZES:
                        num_ngrams = count_ngram(text, sz)
                        counts[sz] += num_ngrams
                        pbar.update()

            # compute estimates
            denom = len(df) if text_col in df.columns else len(sampled)
            for sz in SIZES:
                mean = math.ceil(counts[sz] / denom)
>>>>>>> 8db5542e13743bc15b6e4dbce8cba9484e46e6b6
                estimate_count = mean * DATA_SIZE
                outpath = os.path.join(WORK_DIR, benchmark_tag, f"ngram_count_{sz}.txt")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                with open(outpath, 'w') as f:
                    f.write(str(estimate_count)+"\n")

                print(f"Wrote ngram_count to {outpath}, mean = {mean}")


