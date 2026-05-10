# b-bit MinHashLSH

This folder is a standalone b-bit MinHashLSH implementation for the existing
`dedup` benchmark harness. It does not modify the older algorithms.

Pipeline:

```text
text -> preprocessing -> shingling -> full MinHash signature
     -> b-bit masking -> LSH banding -> candidates -> exact Jaccard verification
```

The only difference from standard MinHashLSH is the b-bit masking step:

```python
mask = (1 << b_bits) - 1
bbit_value = hash_value & mask
```

The b-bit index is implemented here as a custom dictionary:

```text
(band_id, stable_band_hash) -> [doc_id, ...]
```

It does not use `datasketch.MinHashLSH` for the b-bit index. Band hashes are
computed with `hashlib.blake2b(band.tobytes())`, and compressed signatures are
stored with the smallest practical dtype:

- `b_bits <= 8`: `uint8`
- `b_bits <= 16`: `uint16`
- `b_bits <= 32`: `uint32`

Run a small example:

```bash
cd LSH-benchmark/dedup/minhash_bbit_lsh
python example.py
```

Run on the benchmark harness from `LSH-benchmark`:

```bash
cd LSH-benchmark
python dedup/minhash_bbit_lsh/bbit_lsh.py \
  --input test_p_0.1 \
  --threshold 0.8 \
  --num-perm 128 \
  --b-bits 8 \
  --num-bands 16 \
  --rows-per-band 8 \
  --shingle-size 1 \
  --max-bucket-size 10000
```

Inputs are expected at:

```text
benchmark_dfs/<input>.jsonl
benchmark_dfs/<input>.csv
```

Outputs are written to:

```text
<input>/minhash_bbit_lsh_results/
```

The runner writes predictions, harness score metrics, and additional stats:

- candidate pair count
- runtime split for signature/query/insert
- estimated full signature bytes
- estimated b-bit signature bytes
- bucket count and entry count in the custom index
- estimated custom index bytes
- peak Python memory if available from `tracemalloc`
- candidate Jaccard histogram
