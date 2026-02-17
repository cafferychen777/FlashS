"""
Expression matrix normalization utilities.

Handles normalization for sparse expression matrices while
maintaining sparsity structure where possible.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


ArrayLike = Union[NDArray[np.floating], sparse.spmatrix]


def normalize_total(
    X: ArrayLike,
    target_sum: float | None = None,
    inplace: bool = False,
) -> ArrayLike:
    """
    Normalize each cell to have the same total counts.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix (can be sparse).
    target_sum : float, optional
        Target sum for each cell. If None, uses median of total counts.
    inplace : bool, default=False
        Whether to modify X in place (only for dense arrays).

    Returns
    -------
    X_norm : array-like
        Normalized expression matrix.
    """
    is_sparse = sparse.issparse(X)

    # Compute cell totals
    if is_sparse:
        totals = np.asarray(X.sum(axis=1)).ravel()
    else:
        totals = X.sum(axis=1)

    # Default target: median of non-zero cell totals
    if target_sum is None:
        nonzero_totals = totals[totals > 0]
        if len(nonzero_totals) == 0:
            return X.copy() if not inplace else X
        target_sum = float(np.median(nonzero_totals))

    # Compute scaling factors
    totals = np.maximum(totals, 1e-10)  # Avoid division by zero
    scale_factors = target_sum / totals

    if is_sparse:
        # Sparse: element-wise row scaling, no intermediate matrix
        return X.multiply(scale_factors[:, np.newaxis])
    else:
        if inplace:
            X *= scale_factors[:, np.newaxis]
            return X
        else:
            return X * scale_factors[:, np.newaxis]


def log1p_transform(
    X: ArrayLike,
    base: float | None = None,
) -> ArrayLike:
    """
    Apply log(1 + x) transformation.

    Parameters
    ----------
    X : array-like
        Expression matrix.
    base : float, optional
        Logarithm base. If None, uses natural log.

    Returns
    -------
    X_log : array-like
        Log-transformed matrix.
    """
    if sparse.issparse(X):
        X_log = X.copy()
        X_log.data = np.log1p(X_log.data)
        if base is not None:
            X_log.data /= np.log(base)
        return X_log
    else:
        X_log = np.log1p(X)
        if base is not None:
            X_log /= np.log(base)
        return X_log
