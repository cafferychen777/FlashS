# FlashS

**F**requency-domain **L**inearized **A**daptive **S**patial **H**ypothesis testing via **S**ketching

Ultra-fast spatially variable gene (SVG) detection for million-scale spatial transcriptomics.

## Features

- **SVG Detection**: Per-gene complexity `O(nnz * D + nnz * log(nnz))`; sparse projection is `O(nnz * D)`
- **Multi-kernel Cauchy combination**: Binary, rank, and direct tests across multiple bandwidth scales
- **Scalability**: Handles million-scale cells on standard hardware (3.94M cells in 12.6 min, <22 GB RAM)
- **Memory Efficient**: O(D) per gene, no n x n or n x D matrix construction

## Installation

```bash
git clone https://github.com/cafferychen777/FlashS.git
cd FlashS
pip install -e .
```

Optional extras:

```bash
# AnnData I/O
pip install -e ".[io]"

# Full stack (AnnData + visualization)
pip install -e ".[full]"

# Development (testing)
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from scipy import sparse
from flashs import FlashS

# Simulated data
rng = np.random.default_rng(0)
n_cells, n_genes = 2000, 500
coords = rng.normal(size=(n_cells, 2))
X = sparse.random(n_cells, n_genes, density=0.03, random_state=0, format="csc")

# Fit + test
result = FlashS().fit(coords).test(X)

# Results
print(f"P-values shape: {result.pvalues.shape}")
print(f"Significant genes (q < 0.05): {result.significant_genes(q_threshold=0.05)}")
print(f"Effect sizes: {result.effect_size[:5]}")
```

### With AnnData

```python
from flashs.io import run_flashs

# Requires: pip install -e ".[io]"
result = run_flashs(adata, spatial_key="spatial")
# Results are stored in adata.var and adata.uns
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a complete walkthrough.

## Method Overview

1. **RFF Mapping**: Transform coordinates to frequency domain via multi-scale Gaussian kernels
2. **Sparse Sketching**: Compute test statistics using only non-zero entries in a single fused pass
3. **Three-Part Test**: Binary (presence), rank (intensity order), and direct (raw value) spatial tests
4. **Multi-Kernel Cauchy Combination**: Combine per-scale p-values across all test types
5. **Analytic P-values**: Satterthwaite chi-squared approximation under null

See [docs/methods.md](docs/methods.md) for full mathematical details.

## Benchmark Reproduction

FlashS achieves state-of-the-art accuracy on the [Open Problems SVG benchmark](https://openproblems.bio/results/spatially_variable_genes/) (mean Kendall tau = 0.936 across 50 datasets).

See [`benchmarks/README.md`](benchmarks/README.md) for instructions on reproducing results.

## Project Structure

```
FlashS/
├── flashs/                 # Main package
│   ├── core/              # Core algorithms (RFF, sketching, p-values, bandwidth)
│   ├── model/             # SVG model (FlashS, FlashSResult)
│   ├── preprocessing/     # Normalization utilities
│   └── io/                # AnnData integration
├── tests/                 # Unit and integration tests
├── examples/              # Quickstart notebook
├── benchmarks/            # Benchmark reproduction scripts
├── open_problems/         # Open Problems benchmark entry
└── docs/                  # Methods documentation
```

## Citation

If you use FlashS in your research, please cite:

```bibtex
@article{yang2026flashs,
  title={FlashS: Ultra-fast Spatially Variable Gene Detection
         via Random Fourier Features and Sparse Sketching},
  author={Yang, Chen and Zhang, Xianyang and Chen, Jun},
  year={2026},
  journal={Manuscript submitted},
  url={https://github.com/cafferychen777/FlashS}
}
```

## License

MIT License
