# Benchmark Reproduction

Instructions for reproducing the FlashS benchmark results reported in the paper.

## Prerequisites

```bash
pip install -e ".[bench]"
```

## Open Problems SVG Benchmark (Table 1 / Fig. 2)

The primary benchmark evaluates FlashS on 50 datasets from the
[Open Problems in Single-Cell Analysis](https://openproblems.bio/results/spatially_variable_genes/)
spatially variable gene detection task.

### Run

```bash
cd benchmarks

# Run FlashS on all 50 datasets
python run_all_benchmarks.py

# Verify results
python verify_benchmark.py
```

### Expected Output

```
Mean Kendall tau: 0.936
Median Kendall tau: 0.968
Datasets passed: 50/50
```

### Multi-Method Comparison

To compare FlashS against other SVG methods (SPARK-X, scBSP, Hotspot, nnSVG):

```bash
python benchmark_openproblems_methods.py
```

This script requires additional R/Python packages for competitor methods.
See the script header for installation instructions.

## Data

Benchmark datasets are automatically downloaded from the Open Problems
data repository on first run. No manual data download is required.
