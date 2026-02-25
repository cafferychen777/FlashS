"""
P-value computation and multiple testing correction.

Provides:
- Cauchy combination for merging kernel-level p-values (single and batch)
- Scaled chi-square p-values for test statistics
- Multiple testing adjustment (BH, Storey, Bonferroni, etc.)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _cauchy_t_to_pvalue(T_cct: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert Cauchy statistic(s) to p-value(s).

    P(Cauchy > T) = 0.5 - arctan(T)/π, with asymptotic approximations
    for |T| > 1e10 to avoid floating-point precision loss.

    Works for both scalar and array inputs (returns same shape).
    """
    T_cct = np.asarray(T_cct)
    is_large_pos = T_cct > 1e10
    is_large_neg = T_cct < -1e10
    # Safe denominator for asymptotic branches: np.where evaluates all
    # branches eagerly, so substitute 1.0 where |T| is small to avoid
    # division-by-zero warnings (the result is discarded by np.where).
    T_safe = np.where(is_large_pos | is_large_neg, T_cct, 1.0)
    pval = np.where(
        is_large_pos,
        1.0 / (np.pi * T_safe),
        np.where(
            is_large_neg,
            1.0 - 1.0 / (np.pi * np.abs(T_safe)),
            0.5 - np.arctan(T_cct) / np.pi,
        ),
    )
    return np.clip(pval, np.finfo(float).tiny, 1.0)


def cauchy_combination(
    pvalues: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """
    Combine P-values using Cauchy combination rule.

    The Cauchy combination test is robust and valid under arbitrary
    dependency structures between tests.

    T_cct = Σ w_i * tan((0.5 - p_i) * π)

    Under H0, T_cct approximately follows standard Cauchy distribution.

    Parameters
    ----------
    pvalues : ndarray
        Array of P-values to combine. Values at boundaries (0 or 1) are
        clipped to [1e-15, 1-1e-15]. NaN/Inf entries are excluded.
    weights : ndarray, optional
        Weights for each P-value (sum to 1). Default: equal weights.

    Returns
    -------
    combined_pvalue : float
        Combined P-value.

    Notes
    -----
    Boundary handling matches ``batch_cauchy_combination``: p-values are
    clipped, not filtered. A p-value of 0 represents the strongest possible
    signal and is preserved (clipped to 1e-15). Filtering it would silently
    discard the most significant evidence.

    References
    ----------
    Liu & Xie (2020). Cauchy Combination Test: A Powerful Test With
    Analytic p-Value Calculation Under Arbitrary Dependency Structures.
    """
    pvalues = np.asarray(pvalues, dtype=float)

    if len(pvalues) == 0:
        return 1.0

    # Exclude genuinely invalid entries (NaN, Inf) before any logic
    valid_mask = np.isfinite(pvalues)
    if not np.any(valid_mask):
        return 1.0

    if np.sum(valid_mask) == 1:
        return float(np.clip(pvalues[valid_mask][0], np.finfo(float).tiny, 1.0))

    p_valid = pvalues[valid_mask]

    if weights is not None:
        w = np.asarray(weights, dtype=float)[valid_mask]
        w_sum = np.sum(w)
        if w_sum == 0.0:
            return 1.0  # zero total weight → no evidence
        w = w / w_sum
    else:
        w = np.ones(len(p_valid)) / len(p_valid)

    # Clip boundaries — same strategy as batch_cauchy_combination
    p_safe = np.clip(p_valid, 1e-15, 1 - 1e-15)
    cauchy_stats = np.tan((0.5 - p_safe) * np.pi)

    T_cct = np.sum(w * cauchy_stats)

    return float(_cauchy_t_to_pvalue(T_cct))


def batch_cauchy_combination(
    pvalues_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Vectorized Cauchy combination across columns (equal weights).

    For each row, combines all kernel-level p-values into a single
    combined p-value via the Cauchy combination test.

    Parameters
    ----------
    pvalues_matrix : ndarray of shape (n_tests, n_kernels)
        P-values from multiple kernels for each test.

    Returns
    -------
    combined : ndarray of shape (n_tests,)
        Combined p-values.
    """
    pvalues_matrix = np.asarray(pvalues_matrix, dtype=float)

    # Mask NaN/Inf entries — consistent with cauchy_combination
    valid_mask = np.isfinite(pvalues_matrix)
    n_valid_per_row = valid_mask.sum(axis=1)
    all_invalid = n_valid_per_row == 0

    p_safe = np.where(valid_mask, np.clip(pvalues_matrix, 1e-15, 1 - 1e-15), 0.0)
    cauchy_stats = np.tan((0.5 - p_safe) * np.pi)

    # Zero out invalid entries so they don't contribute to the sum
    cauchy_stats = np.where(valid_mask, cauchy_stats, 0.0)
    # Mean over valid entries only (safe: all_invalid rows handled below)
    safe_n = np.where(all_invalid, 1, n_valid_per_row)
    T_cct = cauchy_stats.sum(axis=1) / safe_n

    result = _cauchy_t_to_pvalue(T_cct)
    # No valid evidence → cannot reject H0 → p = 1.0
    result[all_invalid] = 1.0
    return result


def adjust_pvalues(
    pvalues: NDArray[np.floating],
    method: Literal["bh", "bonferroni", "holm", "by", "storey", "none"] = "bh",
) -> NDArray[np.floating]:
    """
    Adjust P-values for multiple testing.

    Parameters
    ----------
    pvalues : ndarray
        Raw P-values.
    method : {'bh', 'bonferroni', 'holm', 'by', 'storey', 'none'}, default='bh'
        Correction method:
        - 'bh': Benjamini-Hochberg (FDR control)
        - 'bonferroni': Bonferroni (FWER control)
        - 'holm': Holm-Bonferroni (FWER control, less conservative)
        - 'by': Benjamini-Yekutieli (FDR under dependency)
        - 'storey': Storey's q-value (less conservative than BH, estimates π₀)
        - 'none': No correction (returns raw p-values)

    Returns
    -------
    adjusted : ndarray
        Adjusted P-values (q-values for BH/BY/Storey).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)

    if n == 0:
        return pvalues

    # Handle NaN values
    nan_mask = np.isnan(pvalues)
    valid_pvalues = pvalues[~nan_mask]
    n_valid = len(valid_pvalues)

    # Validate p-values are in [0, 1] (after NaN exclusion)
    if n_valid > 0:
        pmin, pmax = valid_pvalues.min(), valid_pvalues.max()
        if pmin < 0.0 or pmax > 1.0:
            raise ValueError(
                f"P-values must be in [0, 1], got range [{pmin:.6g}, {pmax:.6g}]"
            )

    if n_valid == 0:
        return pvalues

    if method == "none":
        # No correction - return raw p-values
        adjusted_valid = valid_pvalues.copy()

    elif method == "bonferroni":
        adjusted_valid = np.minimum(valid_pvalues * n_valid, 1.0)

    elif method == "holm":
        # Holm step-down procedure
        order = np.argsort(valid_pvalues)
        adjusted_valid = np.zeros(n_valid)

        for i, idx in enumerate(order):
            adjusted_valid[idx] = valid_pvalues[idx] * (n_valid - i)

        # Enforce monotonicity
        adjusted_valid[order] = np.maximum.accumulate(adjusted_valid[order])
        adjusted_valid = np.minimum(adjusted_valid, 1.0)

    elif method in ("bh", "by"):
        # Benjamini-Hochberg or Benjamini-Yekutieli
        order = np.argsort(valid_pvalues)
        ranks = np.arange(1, n_valid + 1)

        if method == "by":
            # BY includes correction factor c(m) = sum(1/i)
            cm = np.sum(1.0 / ranks)
            adjusted_valid = valid_pvalues[order] * n_valid * cm / ranks
        else:
            adjusted_valid = valid_pvalues[order] * n_valid / ranks

        # Enforce monotonicity (from largest to smallest)
        adjusted_valid = np.minimum.accumulate(adjusted_valid[::-1])[::-1]
        adjusted_valid = np.minimum(adjusted_valid, 1.0)

        # Restore original order
        adjusted_valid_orig = np.zeros(n_valid)
        adjusted_valid_orig[order] = adjusted_valid
        adjusted_valid = adjusted_valid_orig

    elif method == "storey":
        # Storey's q-value method (less conservative than BH)
        # Estimates π₀ (proportion of true nulls) to get tighter FDR control

        # Estimate π₀ using the smoother method
        lambdas = np.arange(0.05, 0.95, 0.05)
        pi0_estimates = np.array([
            np.mean(valid_pvalues > lam) / (1 - lam) for lam in lambdas
        ])
        # Use smoothed estimate from high lambda values
        pi0 = min(1.0, np.mean(pi0_estimates[-5:]))

        # Sort p-values
        order = np.argsort(valid_pvalues)
        pvalues_sorted = valid_pvalues[order]
        ranks = np.arange(1, n_valid + 1)

        # Compute q-values: q(p) = min_{t>=p} π₀ * m * p / #{p_j <= p}
        adjusted_sorted = pi0 * n_valid * pvalues_sorted / ranks

        # Enforce monotonicity (from largest to smallest)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        # Restore original order
        adjusted_valid = np.zeros(n_valid)
        adjusted_valid[order] = adjusted_sorted

    else:
        raise ValueError(f"Unknown method: {method}")

    # Restore NaN positions
    adjusted = np.full(n, np.nan)
    adjusted[~nan_mask] = adjusted_valid

    return adjusted
