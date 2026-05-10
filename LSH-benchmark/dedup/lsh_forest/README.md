# MinHash LSH Forest

This folder adds a separate LSH Forest deduper without modifying the existing
deduplication algorithms.

Pipeline:

```text
text -> preprocessing -> shingles -> MinHash signature -> LSH Forest top-k candidates -> Jaccard verification
```

Run one benchmark:

```powershell
cd D:\Project\LSHBloom-Spark-Deduplication\LSH-benchmark
$env:PYTHONUTF8='1'
python dedup/lsh_forest/lsh_forest.py --input test_p_0.1 --threshold 0.8 --num-perm 128 --num-trees 8 --top-k 50 --shingle-size 1
```

Outputs are written to:

```text
test_p_0.1/lsh_forest_results/
```

`MinHashLSHForest` returns approximate top-k candidates instead of threshold
buckets, so this runner uses `--top-k` for candidate generation and then applies
exact Jaccard verification with `--threshold`.

Run a threshold sweep:

```powershell
python run_lsh_forest_psweep.py --datasets 0.1 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --num-perm 128 --num-trees 8 --top-k 50 --shingle-size 1 --summary-out lsh_forest_test_p_0.1_summary.csv
```
