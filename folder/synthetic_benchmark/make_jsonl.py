"""
Make jsonl from benchmark df for dedup harness
"""

import pandas as pd
import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    file_dir = os.path.dirname(args.file)
    file_stem = os.path.splitext(os.path.basename(args.file))[0]
    df = pd.read_csv(args.file, sep='|')
    df = df.sort_values('id')

    outpath = os.path.join(file_dir, f"{file_stem}.jsonl")
    fout = open(outpath, 'w')
    for row in df.itertuples():
        obj = {
            'id': row.id,
            'text': row.new_text
        }

        json.dump(obj, fout)
        fout.write('\n')

    fout.close()
    print(f"Written to {outpath}")

    