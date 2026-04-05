# MinHashLSH Benchmark - Cloud Deployment Package

## Overview
This package contains all necessary files to run the MinHashLSH deduplication benchmark on a cloud platform.

## Files Included

### Data Files (`data/`)
- **arxiv_html_reference_v2_neardup.jsonl** (92 MB) - HTML parsed from arXiv, filtered for quality
- **pymupdf.jsonl** (223 MB) - Text extracted using PyMuPDF from PDFs
- **pypdf.jsonl** (221 MB) - Text extracted using pypdf library from PDFs
- **arxiv_tesseract_ocr_all.jsonl** (209 MB) - Text extracted using Tesseract OCR

Total: ~745 MB

### Code Files
- **minhashlsh-benchmark.ipynb** - Main benchmark notebook with parser-centric sampling and LSH evaluation
- **requirements.txt** - Python dependencies

### Output Directory
- **output/** - Directory where results will be saved (auto-generated during run)
  - `tuning_results.csv` - Tuning phase metrics (30 rows)
  - `test_results.csv` - Test phase metrics (6 rows)
  - `best_config.json` - Optimal hyperparameters

## Setup Instructions

### Local Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   - In Jupyter: Open `minhashlsh-benchmark.ipynb` and execute cells in order
   - In Colab: Upload files, modify paths (see below)

### Cloud Execution (Kaggle/Colab/AWS)
1. Upload the entire `benchmark_cloud_package/` directory to your cloud platform
2. Modify **Cell 2** (Configuration section) to use relative paths:
   ```python
   # Change FROM:
   # _ABS = r"d:\Project\LSHBloomExperiments\notebook"
   # 
   # Change TO:
   _ABS = os.path.dirname(os.path.abspath(__file__))
   # OR for Colab/Kaggle:
   # _ABS = "/path/to/benchmark_cloud_package"
   ```

3. Execute all cells in order

## Configuration Details

### Benchmark Parameters (Cell 3)
- **TUNING_PREVALENCES**: [0.3, 0.5, 0.7] - Duplicate proportions for hyperparameter tuning
- **TEST_PREVALENCES**: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] - Prevalences for testing
- **THRESHOLD_GRID**: [0.5-0.9] - LSH similarity thresholds
- **NUM_PERM_GRID**: [64, 128, 256] - Number of permutations for MinHash
- **NGRAM_GRID**: [1, 3] - Word n-gram sizes

### Benchmark Design (Cell 7)
- **Parser-centric originals**: Balanced assignment across parsers
- **Parser-duplication**: Same doc_id, different parser (mod=1)
- **Truncation-duplication**: Random 1-20% text removal (mod=2)
- **Stream evaluation**: Mimics real LSHDeduper behavior

## Output Format

### tuning_results.csv
Columns: `ngram_n, num_perm, threshold, prevalence, precision, recall, f1`
- 30 rows (2 ngram × 3 num_perm × 5 threshold × 3 prevalence)

### test_results.csv
Columns: `prevalence, precision, recall, f1` 
- 6 rows (one per test prevalence)

### best_config.json
```json
{
  "ngram_n": 1 or 3,
  "num_perm": 64/128/256,
  "threshold": 0.5-0.9,
  "mean_test_f1": 0.XX,
  "mean_test_precision": 0.XX,
  "mean_test_recall": 0.XX
}
```

## Platform-Specific Notes

### Kaggle Notebooks
- Upload as dataset, reference path: `/kaggle/input/benchmark_cloud_package/`
- Adjust Cell 2: `_ABS = "/kaggle/input/benchmark_cloud_package"`
- Output will go to `/kaggle/working/minhashlsh_results/`

### Google Colab
- Use `!unzip benchmark_cloud_package.zip` or mount Google Drive
- Adjust Cell 2 accordingly
- Can download results back after execution

### AWS SageMaker
- Upload to S3, configure SageMaker notebook instance
- Create symlinks in notebook environment to data folder
- Results can be saved back to S3

## Troubleshooting

**Issue**: "FileNotFoundError" when reading JSONL files
- **Solution**: Check Cell 2 paths. Verify all .jsonl files are in `data/` folder

**Issue**: Low precision/recall on parser-dups (mod=1)
- **Expected behavior**: Parser format differences (HTML vs PyMuPDF plain text) cause legitimate LSH misses
- **Analysis**: Keep mod=1 and mod=2 results separate in evaluation

**Issue**: Out of memory
- **Solution**: Reduce dataset size by filtering JSONL files before upload
- **Alternative**: Run on cloud with more RAM (SageMaker, Colab Pro)

## Expected Runtime
- Local machine (4 cores): ~15-30 minutes
- Cloud (8+ cores): ~5-15 minutes
- Full tuning + test on all 6 test prevalences

## Questions?
Refer to the paper or comments in the notebook for detailed explanations of each cell.
