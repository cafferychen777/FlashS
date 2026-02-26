# FlashS

[![Tests](https://github.com/cafferychen777/FlashS/actions/workflows/test.yml/badge.svg)](https://github.com/cafferychen777/FlashS/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/flashs)](https://pypi.org/project/flashs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/flashs)](https://pypi.org/project/flashs/)

**F**requency-domain **L**inearized **A**daptive **S**patial **H**ypothesis testing via **S**ketching

Ultra-fast spatially variable gene (SVG) detection for spatial transcriptomics at any scale.

## Features

- **Sub-linear complexity**: `O(nnz · D)` per gene — exploits expression sparsity
- **Multi-kernel Cauchy combination**: Binary, rank, and direct tests across multiple bandwidth scales
- **Atlas-scale**: 3.94M cells in 12.6 min, <22 GB RAM on a single node
- **Memory efficient**: `O(D)` per gene, no `n × n` or `n × D` matrix construction
- **scverse integration**: Scanpy-style `tl`/`pl` API with AnnData support

## Installation

```bash
pip install flashs
```

Optional extras:

```bash
pip install "flashs[full]"    # AnnData + visualization
pip install "flashs[io]"      # AnnData only
pip install "flashs[dev]"     # Testing
```

## Quick Start

### Scanpy-style API (recommended)

```python
import flashs

# adata: AnnData with spatial coordinates in adata.obsm["spatial"]
flashs.tl.spatial_variable_genes(adata)

# Results stored in adata.var and adata.uns
sig = adata.var.query("flashs_qvalue < 0.05")

# Visualization
flashs.pl.spatial_variable_genes(adata)
flashs.pl.volcano(adata)
```

### Standalone API

```python
import numpy as np
from scipy import sparse
from flashs import FlashS

coords = np.random.randn(50000, 2)
X = sparse.random(50000, 1000, density=0.03, format="csc")

result = FlashS().fit(coords).test(X)

print(result.significant_genes())
print(result.to_dataframe().head())
```

## API Reference

### `flashs.tl` — Tools

#### `flashs.tl.spatial_variable_genes(adata, ...)`

Detect spatially variable genes. Results are stored in `adata.var` and `adata.uns`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | Annotated data with `adata.obsm[spatial_key]` |
| `spatial_key` | `str` | `"spatial"` | Key in `adata.obsm` for spatial coordinates |
| `layer` | `str \| None` | `None` | Expression layer; `None` uses `adata.X` |
| `genes` | `list[str] \| None` | `None` | Subset of genes to test; `None` tests all |
| `n_features` | `int` | `500` | Number of Random Fourier Features (D) |
| `n_scales` | `int` | `7` | Number of bandwidth scales (L) |
| `min_expressed` | `int` | `5` | Minimum expressing cells to test a gene |
| `key_added` | `str` | `"flashs"` | Prefix for result columns in `adata.var` |
| `copy` | `bool` | `False` | Return a modified copy instead of updating in place |
| `random_state` | `int \| None` | `0` | Random seed for reproducibility |

**Output columns** in `adata.var`:

| Column | Description |
|--------|-------------|
| `{key}_pvalue` | Raw p-values (Cauchy combination across all kernels) |
| `{key}_qvalue` | FDR-adjusted q-values (Benjamini-Hochberg) |
| `{key}_statistic` | Combined test statistics |
| `{key}_effect_size` | Spatial effect size (observed / expected statistic ratio) |
| `{key}_pvalue_binary` | P-values from binary (presence/absence) test |
| `{key}_pvalue_rank` | P-values from rank-transformed test |
| `{key}_n_expressed` | Number of expressing cells per gene |

**Metadata** in `adata.uns[key]`:

| Key | Description |
|-----|-------------|
| `n_tested` | Number of genes tested |
| `n_significant` | Number of significant genes (q < 0.05) |
| `spatial_key` | Spatial coordinates key used |
| `n_features` | RFF features used |
| `n_scales` | Number of bandwidth scales |
| `bandwidths` | Bandwidth values used |

### `flashs.pl` — Plotting

#### `flashs.pl.spatial_variable_genes(adata, ...)`

Plot top spatially variable genes on spatial coordinates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | AnnData with Flash-S results |
| `key` | `str` | `"flashs"` | Key prefix from `tl.spatial_variable_genes` |
| `n_top` | `int` | `6` | Number of top genes to plot |
| `spot_size` | `float \| None` | `None` | Scatter point size; `None` auto-detects |
| `ncols` | `int` | `3` | Columns in subplot grid |
| `cmap` | `str` | `"viridis"` | Colormap |
| `figsize` | `tuple \| None` | `None` | Figure size; `None` auto-computes |
| `save` | `str \| None` | `None` | Path to save figure |
| `show` | `bool` | `True` | Whether to call `plt.show()` |

#### `flashs.pl.volcano(adata, ...)`

Volcano plot (-log10 p-value vs effect size).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | AnnData with Flash-S results |
| `key` | `str` | `"flashs"` | Key prefix |
| `q_threshold` | `float` | `0.05` | Significance threshold |
| `effect_threshold` | `float` | `0.0` | Minimum effect size |
| `n_label` | `int` | `10` | Number of top genes to label |
| `figsize` | `tuple` | `(5, 4)` | Figure size |
| `save` | `str \| None` | `None` | Path to save figure |
| `show` | `bool` | `True` | Whether to call `plt.show()` |

### `flashs.FlashS` — Core Model

#### `FlashS(n_features=500, n_scales=7, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_features` | `int` | `500` | Random Fourier Features (D) |
| `n_scales` | `int` | `7` | Bandwidth scales (L) |
| `kernel` | `KernelType` | `GAUSSIAN` | Kernel type for RFF |
| `bandwidth` | `float \| list \| None` | `None` | Manual bandwidths; `None` uses adaptive selection |
| `min_expressed` | `int` | `5` | Minimum expressing cells |
| `normalize` | `bool \| str` | `False` | Library-size normalization |
| `log_transform` | `bool` | `False` | log1p transformation |
| `adjustment` | `str` | `"bh"` | Multiple testing correction (`"bh"`, `"bonferroni"`, `"holm"`, `"by"`, `"storey"`, `"none"`) |
| `random_state` | `int \| None` | `0` | Random seed |

#### `FlashS.fit(coords) -> FlashS`

Fit the model to spatial coordinates. Precomputes RFF parameters and null distribution statistics.

#### `FlashS.test(X, gene_names=None, return_projections=False) -> FlashSResult`

Test genes for spatial variability. Operates in `O(nnz · D)` per gene.

#### `FlashS.fit_test(coords, X, gene_names=None, ...) -> FlashSResult`

Convenience method combining `fit()` and `test()`.

### `FlashSResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `pvalues` | `ndarray` | Raw p-values |
| `qvalues` | `ndarray` | FDR-adjusted q-values |
| `statistics` | `ndarray` | Test statistics |
| `effect_size` | `ndarray` | Spatial effect size |
| `pvalues_binary` | `ndarray` | Binary test p-values |
| `pvalues_rank` | `ndarray` | Rank test p-values |
| `n_expressed` | `ndarray` | Expressing cells per gene |
| `gene_names` | `list[str]` | Gene names |
| `n_tested` | `int` | Genes tested |
| `n_significant` | `int` | Significant genes (q < 0.05) |
| `tested_mask` | `ndarray \| None` | Boolean mask of tested genes |
| `projections` | `ndarray \| None` | RFF projections (if requested) |

| Method | Description |
|--------|-------------|
| `significant_genes(q_threshold=0.05)` | List of significant gene names |
| `to_dataframe()` | Convert to pandas DataFrame sorted by p-value |
| `get_spatial_embedding(genes=None)` | L2-normalized RFF projections for clustering |

### `flashs.io.run_flashs(adata, ...)`

Legacy convenience wrapper. Returns `FlashSResult` (inplace) or modified `AnnData` (copy).

## Method

1. **RFF Mapping**: Multi-scale Gaussian kernel approximation via Random Fourier Features
2. **Sparse Sketching**: Test statistics computed from non-zero entries only — `O(nnz · D)` per gene
3. **Three-Part Test**: Binary (presence), rank (intensity order), direct (raw value)
4. **Cauchy Combination**: Merge per-scale p-values across all test types under arbitrary dependency
5. **Analytic P-values**: Satterthwaite chi-squared approximation with kurtosis correction

See [docs/methods.md](docs/methods.md) for full mathematical details.

## Benchmarks

FlashS achieves state-of-the-art accuracy on the [Open Problems SVG benchmark](https://openproblems.bio/results/spatially_variable_genes/) (mean Kendall tau = 0.936 across 50 datasets).

## Citation

```bibtex
@article{yang2026flashs,
  title={Frequency-domain sparse kernel testing enables scalable
         spatially variable gene discovery with controlled inference
         at atlas scale},
  author={Yang, Chen and Zhang, Xianyang and Chen, Jun},
  year={2026},
  journal={Manuscript submitted},
  url={https://github.com/cafferychen777/FlashS}
}
```

## License

MIT License
