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
