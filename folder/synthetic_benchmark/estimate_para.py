"""
Script that gets a probabilistic estimate of the number of paragraphs in a corpus.
Useful for providing this information to deduplication routines that need to set
the size of a Bloom Filter, for instance.
"""

import os
from glob import glob
from typing import List
import math
import pandas as pd
from config import *
from tqdm.autonotebook import tqdm

def count_para(text: str) -> List[str]:
    """
    Count paragraphs in text
    """
    return len(text.split("\n"))

if __name__ == "__main__":
    N = 1000
    count = 0

    benchmark_csv_files = glob(f"{DATA_PATH}/{DATA_TAG}_bench*.csv")

    for benchmark_csv in benchmark_csv_files:
        benchmark_tag = os.path.basename(benchmark_csv).split('.csv')[0]
        df = pd.read_csv(benchmark_csv, sep="|").sample(n=N)
        df = df[["id", "new_text"]]
        with tqdm(total=N, desc=f"({benchmark_tag}) Estimating paragraph counts...") as pbar:
            for row in df.itertuples(index=False):
                count += count_para(row.new_text)
                pbar.update()
        
        mean = math.ceil(count / N)
        estimate_count = mean * DATA_SIZE
        outpath = os.path.join(WORK_DIR, benchmark_tag, f"para_count.txt")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, 'w') as f:
            f.write(str(estimate_count)+"\n")

        print(f"Wrote para_count to {outpath}, mean = {mean}")


