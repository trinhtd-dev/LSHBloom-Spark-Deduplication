import argparse
import os
from pathlib import Path
import random
import numpy as np
from dedup_benchmark_utils import *

OUTPATH = "./benchmark_dfs"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-per-source", help="Num data entries taken from each parser", type=int, default=4000)
    parser.add_argument("-p", help="Proportion of positive labels (control skew) in [0,1]", type=float, required=True)
    parser.add_argument("-o", "--output", help="Output tag for csv", type=str, required=True,)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # FIND PARSER DUPLICATES
    N_per_source = args.N_per_source
    p_src = Path('./')
    parser_sources = ['html', 'nougat', 'pymupdf', 'tesseract']

    # choose per parser duplicate sample limits to ensure approximate 50/50 split between parser and truncation duplicates
    m = (N_per_source * len(parser_sources)) # num expected non duplicate documents
    num_planned_duplicates = int(m * ((1/(1-args.p)) - 1)) # num expected duplicates based on that number
    num_planned_parser_duplicates = num_planned_duplicates // 2
    per_parser_sample_limit = num_planned_parser_duplicates // len(parser_sources) # set per parser limits
    print(f'Per parser sample limit: {per_parser_sample_limit}')

    # Step 1: Sample N_per_source data entries from each parser and collect 40 unique paths
    data_list, sampled_paths_per_parser = sample_parser_data(parser_sources, N_per_source, p_src, per_parser_sample_limit)

    # Step 2: Collect duplicate entries from between parsers, but keep only those that are pairwise unique
    data_dupl_list = collect_duplicates(sampled_paths_per_parser, parser_sources, p_src, per_parser_sample_limit)

    # Check that there are no more than 1 occurrence of any path in the final data
    path_counts = np.unique([d['path'] for d in data_dupl_list], return_counts=True)
    # print("Path counts: ", path_counts[1])
    assert (path_counts[1] == 1).all(), f"Sampled parser paths are not unique: {path_counts[1]}"

    # Final parser duplicate list
    print(f"Final parser duplicate list size: {len(data_dupl_list)}")

    # PLAN TRUNCATION DUPLICATES
    # Calculate n_trunc duplicates to add so that p% of the dataset is duplicates
    # p = (x + y) / (x + y + m) --> y = (x(p-1) + mp) / (1-p)
    n_trunc = int(((len(data_dupl_list) * (args.p - 1)) + (len(data_list) * args.p)) / (1 - args.p))
    assert n_trunc >= 0, "Num trunc duplicates is negative"
    print(f"Trunc size: {n_trunc}")

    d_dupl_paths = [d['path'] for d in data_dupl_list]
    d_paths = [d['path'] for d in data_list]

    # sample
    available_paths = list(set(d_paths).difference(d_dupl_paths))
    print(f"Num available trunc paths: {len(available_paths)}")
    if len(available_paths) < 50:
        available_paths = d_paths
        print(f"Fail safe - relaxing restrictions on available paths, new number of available trunc paths: {len(available_paths)}")
    random.seed(567)
    sampled_paths = random.sample(available_paths, min(n_trunc, len(available_paths)))
    while len(sampled_paths) < n_trunc:
        diff = n_trunc - len(sampled_paths)
        new_paths = random.sample(available_paths, min(diff, len(available_paths)))
        sampled_paths += new_paths
    print(f"Num sampled trunc paths: {len(sampled_paths)}")
    print(f"Num parser duplicates: {len(data_dupl_list)}")

    # truncate data
    data_trunc_list = []
    for p in sampled_paths:
        for d in data_list:
            if d['path'] == p:
                data_trunc_list.append(d)
                break
    d_trunc_paths = [d['path'] for d in data_trunc_list]

    # Step 3: Apply truncation, random shuffle duplicates, give labels using 'path' as the primary unique identifier
    benchmark_df = make_benchmark_dataframe(data_list, data_dupl_list, data_trunc_list)
    ndup = (benchmark_df.is_duplicate == 1).sum()
    nondup = len(benchmark_df) - ndup
    print(f"benchmark num: {len(benchmark_df)}")
    print(f"Num duplicates: {ndup}")
    print(f"Num original: {nondup}")
    print(f"Percent duplicated: {ndup / len(benchmark_df):.2%}")

    print(benchmark_df.is_duplicate.value_counts())

    outpath = os.path.join(OUTPATH, f"{args.output}.csv")
    benchmark_df.to_csv(outpath, sep='|', escapechar='\\', index=None)