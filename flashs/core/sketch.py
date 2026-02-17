"""
Sketching kernels for Flash-S.

Memory-efficient RFF operations without storing the full N×D feature matrix.

Key functions:
- compute_sum_z:     exact centering vector, O(N·D) time, O(D) memory
- _project_triple*:  fused sparse projection, O(nnz·D) time, O(D) memory
- compute_column_variances:      O(M·D) subsampled
- compute_cov_frobenius_per_scale: O(M·D²/L) subsampled, O(M·D/L) memory

Requires Numba for JIT compilation.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(cache=True, fastmath=True, parallel=True)
def _project_triple_2d(
    values_binary: NDArray[np.float64],
    values_rank: NDArray[np.float64],
    values_direct: NDArray[np.float64],
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fused triple projection for 2D coordinates.

    Computes binary, rank, and direct projections in a single pass
    over the nonzero entries. cos(ω·s + b) is evaluated once per
    (feature, cell) pair instead of twice.

    Parameters
    ----------
    values_binary : ndarray (nnz,)
        Binary values (ones).
    values_rank : ndarray (nnz,)
        Rank values.
    values_direct : ndarray (nnz,)
        Raw expression values.
    coords : ndarray (nnz, 2)
        Spatial coordinates of non-zero entries.
    omega : ndarray (D, 2)
        RFF frequency vectors.
    bias : ndarray (D,)
        RFF phase offsets.
    scale : float
        RFF normalization sqrt(2/D).

    Returns
    -------
    v_binary, v_rank, v_direct : ndarray (D,)
        Projection vectors.
    """
    n_features = omega.shape[0]
    nnz = values_binary.shape[0]

    v_binary = np.zeros(n_features, dtype=np.float64)
    v_rank = np.zeros(n_features, dtype=np.float64)
    v_direct = np.zeros(n_features, dtype=np.float64)

    for k in prange(n_features):
        acc_binary = 0.0
        acc_rank = 0.0
        acc_direct = 0.0
        omega_k0 = omega[k, 0]
        omega_k1 = omega[k, 1]
        bias_k = bias[k]

        for i in range(nnz):
            dot = omega_k0 * coords[i, 0] + omega_k1 * coords[i, 1]
            z_ik = scale * np.cos(dot + bias_k)
            acc_binary += values_binary[i] * z_ik
            acc_rank += values_rank[i] * z_ik
            acc_direct += values_direct[i] * z_ik

        v_binary[k] = acc_binary
        v_rank[k] = acc_rank
        v_direct[k] = acc_direct

    return v_binary, v_rank, v_direct


@njit(cache=True, fastmath=True, parallel=True)
def _project_triple(
    values_binary: NDArray[np.float64],
    values_rank: NDArray[np.float64],
    values_direct: NDArray[np.float64],
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fused triple projection for arbitrary dimensions.

    Same as _project_triple_2d but with a generic inner dot product loop.
    """
    n_features = omega.shape[0]
    nnz = values_binary.shape[0]
    n_dims = coords.shape[1]

    v_binary = np.zeros(n_features, dtype=np.float64)
    v_rank = np.zeros(n_features, dtype=np.float64)
    v_direct = np.zeros(n_features, dtype=np.float64)

    for k in prange(n_features):
        acc_binary = 0.0
        acc_rank = 0.0
        acc_direct = 0.0
        omega_k = omega[k]
        bias_k = bias[k]

        for i in range(nnz):
            dot = 0.0
            for j in range(n_dims):
                dot += omega_k[j] * coords[i, j]
            z_ik = scale * np.cos(dot + bias_k)
            acc_binary += values_binary[i] * z_ik
            acc_rank += values_rank[i] * z_ik
            acc_direct += values_direct[i] * z_ik

        v_binary[k] = acc_binary
        v_rank[k] = acc_rank
        v_direct[k] = acc_direct

    return v_binary, v_rank, v_direct


@njit(cache=True, fastmath=True, parallel=True)
def compute_sum_z(
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """
    Compute sum(Z, axis=0) exactly without materializing Z matrix.

    O(N·D) time, O(D) memory. This MUST be exact (not subsampled) because
    the centering correction mean_y · sum_z enters the test statistic
    quadratically: a subsampled estimate with O(N/√M) error per feature
    creates an O(N²/M) bias in T that dominates the null expectation
    O(N) when N >> M.

    Parameters
    ----------
    coords : ndarray (n, d)
        Spatial coordinates.
    omega : ndarray (D, d)
        Frequency vectors.
    bias : ndarray (D,)
        Phase offsets.
    scale : float
        RFF normalization factor sqrt(2/D).

    Returns
    -------
    sum_z : ndarray (D,)
        Exact sum of Z columns.
    """
    n_cells = coords.shape[0]
    n_features = omega.shape[0]
    n_dims = coords.shape[1]

    sum_z = np.zeros(n_features, dtype=np.float64)

    for k in prange(n_features):
        acc = 0.0
        omega_k = omega[k]
        bias_k = bias[k]
        for i in range(n_cells):
            dot = 0.0
            for j in range(n_dims):
                dot += omega_k[j] * coords[i, j]
            acc += np.cos(dot + bias_k)
        sum_z[k] = scale * acc

    return sum_z


@njit(cache=True, fastmath=True, parallel=True)
def _compute_all_variances(
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """Compute variance of all Z columns in parallel."""
    n = coords.shape[0]
    n_features = omega.shape[0]
    n_dims = coords.shape[1]

    variances = np.zeros(n_features, dtype=np.float64)

    for k in prange(n_features):
        sum_z = 0.0
        sum_z2 = 0.0
        omega_k = omega[k]
        bias_k = bias[k]

        for i in range(n):
            dot = 0.0
            for j in range(n_dims):
                dot += omega_k[j] * coords[i, j]
            z = scale * np.cos(dot + bias_k)
            sum_z += z
            sum_z2 += z * z

        mean = sum_z / n
        variances[k] = sum_z2 / n - mean * mean

    return variances


def compute_column_variances(
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
    max_samples: int = 10000,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """
    Compute variance of each Z column via subsampling (parallelized).

    O(M·D) complexity where M = min(n, max_samples).

    Parameters
    ----------
    coords : ndarray (n, d)
        Spatial coordinates.
    omega : ndarray (D, d)
        Frequency vectors.
    bias : ndarray (D,)
        Phase offsets.
    scale : float
        RFF normalization factor.
    max_samples : int
        Maximum samples for estimation.
    random_state : int, optional
        Random seed.

    Returns
    -------
    variances : ndarray (D,)
        Estimated variance per feature.
    """
    n = coords.shape[0]

    if n == 0:
        return np.zeros(omega.shape[0], dtype=np.float64)

    # Subsample for efficiency
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        coords = coords[idx]

    coords = np.asarray(coords, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    return _compute_all_variances(coords, omega, bias, float(scale))


def compute_cov_frobenius_per_scale(
    coords: NDArray[np.float64],
    omega: NDArray[np.float64],
    bias: NDArray[np.float64],
    scale: float,
    scale_offsets: NDArray[np.int64],
    max_samples: int = 10000,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """
    Compute per-scale ||Cov_z||_F^2 for corrected Satterthwaite variance.

    The standard Satterthwaite approximation assumes independent Z columns,
    but RFF features at large bandwidths are correlated. The corrected
    variance uses ||Cov_z||_F^2 instead of sum(Var(z_k)^2).

    Parameters
    ----------
    coords : ndarray (n, d)
        Spatial coordinates.
    omega : ndarray (D, d)
        Frequency vectors.
    bias : ndarray (D,)
        Phase offsets.
    scale : float
        RFF normalization factor.
    scale_offsets : ndarray (n_scales+1,)
        Cumulative feature counts per scale.
    max_samples : int
        Maximum samples for estimation.
    random_state : int, optional
        Random seed.

    Returns
    -------
    frob_sq : ndarray (n_scales,)
        Per-scale ||Cov_z||_F^2 values.
    """
    n = coords.shape[0]

    # Subsample for efficiency
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        coords_sub = coords[idx]
    else:
        coords_sub = coords

    coords_sub = np.asarray(coords_sub, dtype=np.float64)
    M = coords_sub.shape[0]

    n_scales = len(scale_offsets) - 1
    frob_sq = np.zeros(n_scales)

    if M == 0:
        return frob_sq

    # Stream per-scale: materialize only (M, D_s) instead of full (M, D).
    # Centering is column-wise so per-scale centering is mathematically
    # identical to centering the full matrix then slicing.
    for s in range(n_scales):
        lo, hi = int(scale_offsets[s]), int(scale_offsets[s + 1])
        # Project only this scale's features: (M, D_s)
        Z_s = coords_sub @ omega[lo:hi].T
        Z_s += bias[lo:hi]
        np.cos(Z_s, out=Z_s)
        Z_s *= scale
        Z_s -= Z_s.mean(axis=0)
        # Cov_s = Z_s^T @ Z_s / M, ||Cov_s||_F^2 = ||Z_s^T @ Z_s||_F^2 / M^2
        Gram_s = Z_s.T @ Z_s
        Gram_s /= M
        frob_sq[s] = np.sum(Gram_s * Gram_s)

    return frob_sq
