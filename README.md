# LSHBloom: Optimizing internet-scale text deduplication using probabilistic data structures on Apache Spark

## Description

This project solves the OOM problem when performing text deduplication on a large dataset (peS2o - 39 million scientific papers). We replace the traditional Index (Hashmap) structure of the MinHashLSH algorithm with a probabilistic array **Bloom Filter**, combined with distributed processing on the **Apache Spark** platform.

## Directory structure

- `data/`: Contains raw data files and output results (already gitignored).
- `notebooks/`: Contains Jupyter Notebook files for testing and prototyping the processing flow.
- `src/`: Main source code of the project.
  - `data_prep.py`: Preprocessing text, Shingling (N-grams).
  - `baseline_lsh.py`: Original MinHashLSH algorithm for comparison.
  - `custom_lshbloom.py`: Optimized LSHBloom algorithm.

## Setup Instructions

Run the following commands to initialize the environment and download the dataset:

```bash
chmod +x setup.sh
./setup.sh
source ~/.bashrc
