"""
Script that gets a probabilistic estimate of the number of n-grams in a corpus.
Useful for providing this information to deduplication routines that need to set
the size of a Bloom Filter, for instance.
"""

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
                estimate_count = mean * DATA_SIZE
                outpath = os.path.join(WORK_DIR, benchmark_tag, f"ngram_count_{sz}.txt")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                with open(outpath, 'w') as f:
                    f.write(str(estimate_count)+"\n")

                print(f"Wrote ngram_count to {outpath}, mean = {mean}")


