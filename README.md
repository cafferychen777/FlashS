# FlashS

[![Tests](https://github.com/cafferychen777/FlashS/actions/workflows/test.yml/badge.svg)](https://github.com/cafferychen777/FlashS/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/flashs)](https://pypi.org/project/flashs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/flashs)](https://pypi.org/project/flashs/)

**F**requency-domain **L**inearized **A**daptive **S**patial **H**ypothesis testing via **S**ketching

Frequency-domain kernel testing for spatially variable gene (SVG) detection in sparse spatial transcriptomics.

## Highlights

- **Sparse-aware computation**: `O(nnz · D)` per gene with `O(D)` per-gene memory; no dense `n × n` kernel matrix
- **Robust evidence aggregation**: Binary, rank, and direct tests combined across adaptive kernel scales
- **Atlas-scale practicality**: Benchmarked on 3.94M cells in 12.6 min using 21.5 GB RAM
- **AnnData-native workflow**: Scanpy-style `tl` API plus a standalone model interface

## Installation

```bash
pip install flashs
```

Optional extras:

```bash
pip install "flashs[io]"      # AnnData support
pip install "flashs[dev]"     # Testing
```

## Quick Start

### Scanpy-style API (recommended)

```python
import flashs

# adata: AnnData with spatial coordinates in adata.obsm["spatial"]
flashs.tl.svg(adata)

# Results stored in adata.var and adata.uns
sig = adata.var.query("flashs_qvalue < 0.05")
```

### Standalone API

```python
import numpy as np
from scipy import sparse
from flashs import FlashS

coords = np.random.randn(50000, 2)
X = sparse.random(50000, 1000, density=0.03, format="csc")

result = FlashS().fit_test(coords, X)

print(result.significant_genes())
print(result.to_dataframe().head())
```

## Core API

### AnnData workflow

`flashs.tl.svg(adata, spatial_key="spatial", layer=None, genes=None, n_features=500, n_scales=7, min_expressed=5, key_added="flashs", copy=False, random_state=0)`

This is the recommended entry point for Scanpy and scverse users. It stores per-gene results in `adata.var` and run metadata in `adata.uns[key_added]`.

Key outputs in `adata.var`:

- `{key}_pvalue`
- `{key}_qvalue`
- `{key}_statistic`
- `{key}_effect_size`
- `{key}_pvalue_binary`
- `{key}_pvalue_rank`
- `{key}_n_expressed`

### Standalone workflow

`FlashS(...).fit_test(coords, X, gene_names=None, return_projections=False) -> FlashSResult`

Use the standalone interface when you work outside AnnData. The returned `FlashSResult` provides:

- `significant_genes(...)` to extract discoveries
- `to_dataframe()` to inspect ranked results
- `get_spatial_embedding(...)` to access projection-based embeddings when `return_projections=True`

## Method

FlashS approximates multi-scale spatial kernels with Random Fourier Features, computes sparse sketches from non-zero entries only, and evaluates binary, rank, and direct expression patterns across scales. Analytic p-value approximations and Cauchy combination avoid dense kernel operations while keeping inference tractable on large sparse datasets.

See [docs/methods.md](docs/methods.md) for full mathematical details.

## Benchmarks

In our evaluation on the [Open Problems SVG benchmark](https://openproblems.bio/results/spatially_variable_genes/), FlashS achieved a mean Kendall tau of 0.935 across 50 datasets.

FlashS is intended for settings where informative gene ranking, calibrated inference, and sparse-data scalability all matter.

## Citation

```bibtex
@article{yang2026flashs,
  title={Frequency-domain kernels enable atlas-scale detection of
         spatially variable genes},
  author={Yang, Chen and Zhang, Xianyang and Chen, Jun},
  year={2026},
  journal={Manuscript submitted},
  url={https://github.com/cafferychen777/FlashS}
}
```

## License

MIT License
