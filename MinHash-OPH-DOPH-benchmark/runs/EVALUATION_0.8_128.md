# Evaluation: MinHashLSH + OPH vs MinHashLSH + DOPH

Command used:

```cmd
python MinHash-OPH-DOPH-benchmark\run_benchmark.py --all --sketch oph_doph --sim-threshold 0.8 --num-perm 128 --ngram 1 --workers 4 --force
```

Data:

```text
D:\LEARN\LSHBloom-Spark-Deduplication\LSH-benchmark\benchmark_dfs
```

Run status:

```text
Datasets: test_p_0.1 ... test_p_0.9
Methods: MinHashLSH + OPH, MinHashLSH + DOPH
Jobs completed: 18/18
Missing predictions: 0 rows for every full run
Summary CSV: MinHash-OPH-DOPH-benchmark\runs\summary_0.8_128.csv
```

Average metrics across all 9 datasets:

| method | precision | recall | f1 | acc | bal_acc | avg wall_sec | avg empty bins/doc | avg docs with empty bins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MinHashLSH + OPH | 0.726976 | 0.401607 | 0.510919 | 0.628947 | 0.589721 | 30.180223 | 0.002484 | 6.666667 |
| MinHashLSH + DOPH | 0.726976 | 0.401607 | 0.510919 | 0.628947 | 0.589721 | 30.280583 | 0.002484 | 6.666667 |

Observation:

The OPH and DOPH results are identical on this benchmark run. The diagnostic
columns confirm why: with `num_perm=128`, nearly every document fills every OPH
bin. Across datasets, only about 2-13 documents have any empty bin at all, and
the average number of empty bins per document is only about 0.002-0.004.

That means DOPH's densification step is almost never activated. On this data,
the benchmark is valid for showing runtime/storage behavior, but it is not a
good benchmark for demonstrating a quality difference between OPH and DOPH.

Use the detailed CSV files for reporting:

```text
summary_0.8_128.csv
test_p_*/minhashlsh+oph/oph_0.8_128_score.csv
test_p_*/minhashlsh+oph/oph_0.8_128_stats.csv
test_p_*/minhashlsh+doph/doph_0.8_128_score.csv
test_p_*/minhashlsh+doph/doph_0.8_128_stats.csv
```

Important comparison note:

These OPH/DOPH runs should not be directly claimed as a comparison against
LSHBloom. The original `LSH-benchmark/dedup/lsh/lsh_bloom.py` uses standard
datasketch `MinHash` signatures and changes the index from `MinHashLSH`
dictionary buckets to Bloom filters. By contrast, this folder changes the
signature construction to OPH/DOPH but still uses `MinHashLSH`.

So the fair comparisons are:

```text
MinHashLSH vs MinHashLSH + OPH vs MinHashLSH + DOPH
MinHashLSH vs LSHBloom
LSHBloom vs LSHBloom + OPH/DOPH, only if implemented separately
```

The current repo's LSHBloom script could not be run locally because
`pybloomfilter` is missing:

```text
ModuleNotFoundError: No module named 'pybloomfilter'
```
