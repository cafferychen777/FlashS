from __future__ import annotations

import numpy as np

from flashs.core.pvalue import adjust_pvalues, batch_cauchy_combination, cauchy_combination


def test_adjust_pvalues_range_and_nan_handling() -> None:
    p = np.array([0.01, 0.2, np.nan, 0.7, 1.0])
    q = adjust_pvalues(p, method="bh")

    assert np.isnan(q[2])
    finite = q[~np.isnan(q)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_adjust_pvalues_bh_monotone_on_sorted_pvalues() -> None:
    p = np.array([0.001, 0.01, 0.1, 0.4, 0.9])
    q = adjust_pvalues(p, method="bh")
    assert np.all(np.diff(q) >= -1e-12)


def test_cauchy_combination_boundary_and_nan_handling() -> None:
    # NaN is excluded; p=0 and p=1 are clipped to [1e-15, 1-1e-15]
    # p=0 (strong signal) and p=1 (no signal) cancel → p ≈ 0.5
    p = np.array([0.0, 1.0, np.nan])
    combined = cauchy_combination(p)
    assert 0.4 < combined < 0.6

    # All NaN → fallback to 1.0
    assert cauchy_combination(np.array([np.nan, np.nan])) == 1.0

    # p=0 alone → strongest possible signal (near 0)
    assert cauchy_combination(np.array([0.0])) < 1e-10

    # Matches batch version behavior
    p_batch = np.array([[0.01, 0.2]])
    single = cauchy_combination(np.array([0.01, 0.2]))
    batch = float(batch_cauchy_combination(p_batch)[0])
    assert abs(single - batch) < 1e-12


def test_batch_cauchy_combination_shape_and_bounds() -> None:
    mat = np.array([
        [0.2, 0.3, 0.9],
        [1e-20, 0.8, 0.4],
        [0.6, 0.6, 0.6],
    ])
    out = batch_cauchy_combination(mat)
    assert out.shape == (3,)
    assert np.all((out > 0.0) & (out <= 1.0))


def test_cauchy_single_nan_returns_one() -> None:
    """Single-element NaN input: no evidence → p = 1.0."""
    assert cauchy_combination(np.array([np.nan])) == 1.0


def test_cauchy_all_nan_consistent() -> None:
    """All-NaN: both single and batch must return 1.0, no warnings."""
    import warnings

    all_nan_single = np.array([np.nan, np.nan])
    all_nan_batch = np.array([[np.nan, np.nan], [0.1, 0.2]])

    assert cauchy_combination(all_nan_single) == 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → test failure
        result = batch_cauchy_combination(all_nan_batch)

    assert result[0] == 1.0      # all-NaN row → 1.0
    assert 0.0 < result[1] < 1.0  # valid row → normal result


def test_batch_single_nan_consistency() -> None:
    """Partial NaN: batch and single give the same result."""
    p = np.array([0.2, np.nan, 0.3])
    single = cauchy_combination(p)
    batch = float(batch_cauchy_combination(p.reshape(1, -1))[0])
    assert abs(single - batch) < 1e-12


def test_cauchy_zero_weights_returns_one() -> None:
    """Zero total weight means no evidence → p = 1.0, no warnings."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = cauchy_combination(np.array([0.1, 0.2]), weights=np.array([0.0, 0.0]))
    assert result == 1.0


def test_adjust_pvalues_rejects_out_of_range() -> None:
    """adjust_pvalues must reject p-values outside [0, 1]."""
    import pytest

    with pytest.raises(ValueError, match=r"P-values must be in \[0, 1\]"):
        adjust_pvalues(np.array([-0.1, 0.2, 0.5]), "bh")

    with pytest.raises(ValueError, match=r"P-values must be in \[0, 1\]"):
        adjust_pvalues(np.array([0.2, 1.5]), "bonferroni")
