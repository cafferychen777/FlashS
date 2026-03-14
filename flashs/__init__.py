"""
Flash-S: Frequency-domain kernel testing for spatially variable genes.

Spatially variable gene detection for sparse spatial transcriptomics.

Features
--------
- O(nnz * D) complexity exploiting expression sparsity
- Random Fourier Features for multi-scale kernel approximation
- Three-part test (binary, rank, direct) for zero-inflation robustness
- Analytic p-values via scaled chi-square distribution
- AnnData/Scanpy ecosystem integration

Quick Start
-----------
>>> from flashs import FlashS
>>> result = FlashS().fit_test(coords, expression_matrix)
>>> print(result.significant_genes())

With AnnData (scanpy-style):

>>> import flashs
>>> flashs.tl.svg(adata)
>>> sig = adata.var.query("flashs_qvalue < 0.05")
"""

__version__ = "0.2.1"

# Main API
from .model import FlashS, FlashSResult

# Scanpy-style submodule
from . import tl

# Enum exposed for FlashS(kernel=KernelType.LAPLACIAN)
from .core.rff import KernelType

__all__ = [
    "__version__",
    "FlashS",
    "FlashSResult",
    "tl",
    "KernelType",
]
