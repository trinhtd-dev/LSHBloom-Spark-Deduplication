# spark_lshbloom

`spark_lshbloom` is a Spark package for approximate text deduplication using
MinHash LSH band keys and mergeable Bloom Filters.

The package exposes two execution modes:

- `mode="lshbloom"`: primary LSHBloom mode. It builds partition-local Bloom
  filters, merges them into a global filter, and broadcasts the global filter
  for read-only duplicate queries.
- `mode="minhash_lsh"`: distributed MinHashLSH-style baseline using
  `explode(band_keys) -> groupBy(band_key)`.

Backward-compatible aliases are supported: `bloom`/`bloom_filter` map to
`lshbloom`, and `banding` maps to `minhash_lsh`.

## Install

From the repository root:

```powershell
pip install -e .\spark_lshbloom
```

## Basic Usage

```python
from spark_lshbloom import SparkLSHBloom

lsh = SparkLSHBloom(threshold=0.85, num_perm=256, mode="lshbloom")

# Production-style usage: fit a reference corpus, then query new documents.
lsh.fit(reference_df, text_col="text", id_col="id")
lsh.save("runs/my_bloom_filter")

lsh = SparkLSHBloom.load("runs/my_bloom_filter")
result_df = lsh.transform(new_df, text_col="text", id_col="id")
unique_new_df = result_df.filter(~result_df["is_duplicate"])
```

## One-shot Deduplication

For experiments:

```python
unique_df = lsh.deduplicate(df, text_col="text", id_col="id")
```

In `mode="lshbloom"`, one-shot deduplication uses chunked incremental processing:
each chunk is queried against Bloom filters built from earlier chunks. This
keeps the LSHBloom idea and avoids fitting on the whole dataset before querying
the same dataset. Duplicates inside the same chunk may not be removed, so use
more chunks for stricter experiments.

For exact distributed LSH bucket baseline:

```python
banding = SparkLSHBloom(threshold=0.85, num_perm=256, mode="minhash_lsh")
unique_df = banding.deduplicate(df, text_col="text", id_col="id")
```

## peS2o Benchmark

The benchmark compares the primary LSHBloom index against a distributed
MinHashLSH band table:

```powershell
python spark_lshbloom\benchmarks\benchmark_pes2o.py `
  --input-path MinHash-OPH-DOPH-benchmark\data\pes2o_50k `
  --out-dir spark_lshbloom\runs\pes2o_benchmark_50k `
  --sizes 10000,50000 `
  --num-perm 128 `
  --num-bands 32 `
  --shingle-size 5 `
  --max-shingles 2048 `
  --master local[4] `
  --local-io-mode pyarrow
```

The main output is:

```text
spark_lshbloom/runs/pes2o_benchmark_50k/benchmark_results.csv
```

Important benchmark columns:

- `method`: `lshbloom` or `minhash_lsh`.
- `wall_clock_sec`: elapsed time for building the index.
- `index_size_gb`: saved index size on disk.
- `band_entries`: number of LSH band keys represented.
- `driver_rss_delta_gb`: approximate driver memory increase.

For `lshbloom`, `index_size_gb` is the saved Bloom filter size. For
`minhash_lsh`, it is the compressed Parquet size of the explicit band table.

On a local peS2o 50k benchmark with `num_perm=128`, `num_bands=32`, and
`max_shingles=2048`, the prototype showed about `3.1x` faster index
construction and about `7x` smaller index size for `lshbloom` compared with
the explicit distributed `minhash_lsh` band table.

On Windows, use `--local-io-mode python` if Spark/Arrow workers are unstable.

## Important Design Notes

- Broadcast Bloom filters are read-only on Spark executors.
- Distributed updates are handled by partition-local Bloom filters followed by
  driver-side merge.
- Character n-gram shingling is supported by default.
- Pandas UDFs are used for band-key generation when Spark executes the pipeline.
- `max_shingles` can cap per-document shingle work for long documents.

## Outputs and Metrics

Useful columns and metrics for reporting:

- `is_duplicate`: approximate membership result.
- `band_keys`: generated LSH band keys.
- `bloom.num_bits`, `bloom.num_hashes`: Bloom filter configuration.
- Bloom filter file size from `save(path)`.
- Spark job wall-clock time and partition counts.
