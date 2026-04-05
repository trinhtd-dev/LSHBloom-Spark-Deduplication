# Deduplication Reproducibility Experiments

This repository contains experiments and benchmarks for reproducing various text deduplication strategies, including LSHBloom, MinHashLSH, CCNet, DCLM, and Dolma. It includes a synthetic benchmark generator to evaluate these strategies under controlled conditions.

## Setup

### Prerequisites

Ensure you have Python 3.10 and Rust installed (for compiling `pyhash-archive`).

### Setup

1.  **Install General Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Embedded Libraries**:
    This repository contains modified or specific versions of several libraries embedded in `dedup/`. You need to install them:

    *   **CCNet**:
        Follow the instructions in the repository to build CCNet and then:
        ```bash
        pip install dedup/cc_net
        ```

    *   **Datasketch**:
        ```bash
        pip install dedup/lsh/datasketch
        ```

    *   **PyHash Archive** (Requires Rust/Cargo):
        ```bash
        cd dedup/lsh/pyhash-archive
        pip install .
        ```

## Repository Structure

### `reproducibility/`
The root directory for reproducibility experiments.

*   `requirements.txt`: Python dependencies.
*   `README.md`: This file.

### `dedup/`
Contains implementations of different deduplication strategies and the harness for running them.

*   **`dedup_parsing_harness.py`**: Defines the `DedupHarness` abstract base class. All deduplication experiments inherit from this to ensure consistent execution (reading data, deduplicating, scoring results).
*   **`cc_net/`**: Source code for the CCNet library (Facebook Research).
*   **`ccnet/`**: Contains `ccnet.py`, the runner script for CCNet deduplication experiments using the harness.
*   **`dclm/`**: Contains `dclm.py`, the runner script for DCLM (Data-Comp for Language Models) baseline style deduplication using Bloom filters.
*   **`dolma/`**: Contains `dolma.py`, the runner for Dolma deduplication experiments.
*   **`lsh/`**: LSH (Locality Sensitive Hashing) implementations.
    *   `datasketch/`: Embedded `datasketch` library for MinHash and LSH.
    *   `pyhash-archive/`: Embedded `pyhash-archive` library for fast hashing.
    *   `lsh.py`, `lsh_bloom.py`: Runner scripts for  MinHashLSH and LSHBloom.
*   **`writers.py`**: Utility for writing results.

### `synthetic_benchmark/`
Scripts for generating and managing the synthetic benchmark dataset used to evaluate deduplication fidelity.

*   **`gen_dedup_benchmark.py`**: The main script to generate the synthetic benchmark dataset. It creates duplicates by sampling text from different parsers and creating randomly truncated versions.
*   **`config.py`**: Configuration constants (paths, data sizes) for the benchmark.
*   **`estimate_para.py` / `estimate_ngram.py`**: Utilities to estimate paragraph and n-gram counts, which are required for initializing Bloom filters in DCLM and Dolma experiments.
*   **`dedup_benchmark_utils.py`**: Helper functions for the benchmark generator.

## Running Experiments

1.  **Generate Benchmark Data**:
    Run `synthetic_benchmark/gen_dedup_benchmark.py` to create the dataset (JSONL and CSV ground truth).

2.  **Estimate Counts**:
    Run `estimate_para.py` or `estimate_ngram.py` if running DCLM or Dolma to get the expected item counts.

3.  **Run Deduplication**:
    Execute the specific runner script for the method you want to test. Replace `<benchmark_tag>` with your dataset tag (e.g., `50k`).

    *   **CCNet**:
        ```bash
        python dedup/ccnet/ccnet.py --input <benchmark_tag> --sim-threshold 0.2
        ```

    *   **DCLM (Bloom Filter)**:
        ```bash
        # Requires estimate_ngram.py to have been run
        python dedup/dclm/dclm.py --input <benchmark_tag> --sim-threshold 0.2 --ngram-size 5
        ```

    *   **Dolma**:
        ```bash
        # Paragraph mode (requires estimate_para.py)
        python dedup/dolma/dolma.py --input <benchmark_tag> --sim-threshold 0.2

        # N-gram mode (requires estimate_ngram.py)
        python dedup/dolma/dolma.py --input <benchmark_tag> --sim-threshold 0.2 --ngram-mode --ngram-size 5
        ```

    *   **MinHashLSH**:
        *Note: Standard MinHashLSH experiments require a Redis server running on port 6379.*
        ```bash
        python dedup/lsh/lsh.py --input <benchmark_tag> --sim-threshold 0.5 --num-perm 256 --redis-port 6379 --force-compute-minhash
        ```

    *   **LSHBloom**:
        ```bash
        python dedup/lsh/lsh_bloom.py --input <benchmark_tag> --sim-threshold 0.5 --num-perm 256 --force-compute-minhash
        ```
