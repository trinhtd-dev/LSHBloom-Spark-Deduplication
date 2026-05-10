# MinHashLSH + OPH/DOPH Benchmark

Clean experiment folder for comparing:

- Standard `MinHashLSH`
- `MinHashLSH + OPH`
- `MinHashLSH + DOPH`

The benchmark data is read from:

```text
..\LSH-benchmark\benchmark_dfs
```

## Code Layout

```text
common.py          shared dataset loader, scoring, timing, RAM/disk stats
minhash_lsh.py     standard MinHashLSH baseline
oph_lsh.py         MinHashLSH + OPH
doph_lsh.py        MinHashLSH + DOPH
run_benchmark.py   combined runner for one or all methods
runs\              output folder
```

## Run From CMD

Open CMD at the repo root:

```cmd
cd /d D:\LEARN\LSHBloom-Spark-Deduplication
```

Smoke test first, only 100 documents:

```cmd
python MinHash-OPH-DOPH-benchmark\run_benchmark.py --input test_p_0.1 --sketch all --sim-threshold 0.8 --num-perm 128 --ngram 1 --limit 100 --force
```

Run one full dataset:

```cmd
python MinHash-OPH-DOPH-benchmark\run_benchmark.py --input test_p_0.1 --sketch all --sim-threshold 0.8 --num-perm 128 --ngram 1 --force
```

Run all datasets:

```cmd
python MinHash-OPH-DOPH-benchmark\run_benchmark.py --all --sketch all --sim-threshold 0.8 --num-perm 128 --ngram 1 --force
```

Run only OPH and DOPH on all datasets with parallel workers:

```cmd
python MinHash-OPH-DOPH-benchmark\run_benchmark.py --all --sketch oph_doph --sim-threshold 0.8 --num-perm 128 --ngram 1 --workers 4 --force
```

Run methods separately:

```cmd
python MinHash-OPH-DOPH-benchmark\minhash_lsh.py --input test_p_0.1 --sim-threshold 0.8 --num-perm 128 --ngram 1 --force
python MinHash-OPH-DOPH-benchmark\oph_lsh.py --input test_p_0.1 --sim-threshold 0.8 --num-perm 128 --ngram 1 --force
python MinHash-OPH-DOPH-benchmark\doph_lsh.py --input test_p_0.1 --sim-threshold 0.8 --num-perm 128 --ngram 1 --force
```

## Output Layout

Results are written under `MinHash-OPH-DOPH-benchmark\runs`.

```text
MinHash-OPH-DOPH-benchmark\runs\
  test_p_0.1\
    minhashlsh\
      minhash_0.8_128_preds.csv
      minhash_0.8_128_score.csv
      minhash_0.8_128_stats.csv
      minhash_0.8_128_index.pkl
      signatures\128\*.pkl
    minhashlsh+oph\
      oph_0.8_128_preds.csv
      oph_0.8_128_score.csv
      oph_0.8_128_stats.csv
      oph_0.8_128_index.pkl
      signatures\128\*.npy
    minhashlsh+doph\
      doph_0.8_128_preds.csv
      doph_0.8_128_score.csv
      doph_0.8_128_stats.csv
      doph_0.8_128_index.pkl
      signatures\128\*.npy
```

Use `*_score.csv` for quality metrics:

```text
precision, recall, f1, auc_roc, acc, bal_acc, tp, fp, fn
```

Use `*_stats.csv` for scale metrics:

```text
wall_sec, signature_sec, query_sec, insert_sec, peak_rss_gb,
signature_cache_gb, index_pickle_gb, avg_empty_bins, docs_with_empty_bins,
max_empty_bins
```

The combined runner also writes a summary file:

```text
MinHash-OPH-DOPH-benchmark\runs\summary_0.8_128.csv
```
