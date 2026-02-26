"""
Flash-S: Fast Spatial Variable Gene Detection via Random Fourier Features

Ultra-fast spatial expression pattern detection for million-scale
spatial transcriptomics data.

Features
--------
- O(nnz * D) complexity exploiting expression sparsity
- Random Fourier Features for kernel approximation
- Three-part test (binary, rank, direct) for zero-inflation robustness
- Analytic p-values via scaled chi-square distribution
- AnnData/Scanpy ecosystem integration

Quick Start
-----------
>>> from flashs import FlashS
>>> result = FlashS().fit(coords).test(expression_matrix)
>>> print(result.significant_genes())

With AnnData (scanpy-style):

>>> import flashs
>>> flashs.tl.spatial_variable_genes(adata)
>>> sig = adata.var.query("flashs_qvalue < 0.05")
"""

__version__ = "0.1.1"

# Main API
from .model import (
    FlashS,
    FlashSResult,
)

# Scanpy-style submodules
from . import tl, pl

# Utilities (for advanced users building custom pipelines)
from .core.pvalue import adjust_pvalues, cauchy_combination
from .core.rff import KernelType

__all__ = [
    "__version__",
    # Main API
    "FlashS",
    "FlashSResult",
    # Scanpy-style submodules
    "tl",
    "pl",
    # Utilities
    "KernelType",
    "adjust_pvalues",
    "cauchy_combination",
]


def __getattr__(name: str):
    """Lazy import for optional AnnData integration."""
    if name == "run_flashs":
        from .io import run_flashs
        return run_flashs
    raise AttributeError(f"module 'flashs' has no attribute '{name}'")
