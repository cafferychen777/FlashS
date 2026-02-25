from __future__ import annotations

import numpy as np

from flashs.core.sketch import (
    _project_triple,
    _project_triple_2d,
    compute_column_variances,
    compute_cov_frobenius_per_scale,
    compute_sum_z,
)


def _explicit_triple_projection(
    values_binary: np.ndarray,
    values_rank: np.ndarray,
    values_direct: np.ndarray,
    coords: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = scale * np.cos(coords @ omega.T + bias)
    return values_binary @ Z, values_rank @ Z, values_direct @ Z


def test_project_triple_2d_matches_explicit_projection() -> None:
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(12, 2))
    omega = rng.normal(size=(8, 2))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=8)
    scale = np.sqrt(2.0 / omega.shape[0])
    values_binary = np.ones(12, dtype=np.float64)
    values_rank = rng.normal(size=12)
    values_direct = rng.gamma(shape=2.0, scale=1.0, size=12)

    got = _project_triple_2d.py_func(
        values_binary, values_rank, values_direct, coords, omega, bias, scale
    )
    expected = _explicit_triple_projection(
        values_binary, values_rank, values_direct, coords, omega, bias, scale
    )
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[2], expected[2], rtol=1e-12, atol=1e-12)


def test_project_triple_nd_matches_explicit_projection() -> None:
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(15, 3))
    omega = rng.normal(size=(7, 3))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=7)
    scale = np.sqrt(2.0 / omega.shape[0])
    values_binary = np.ones(15, dtype=np.float64)
    values_rank = rng.normal(size=15)
    values_direct = rng.normal(size=15)

    got = _project_triple.py_func(
        values_binary, values_rank, values_direct, coords, omega, bias, scale
    )
    expected = _explicit_triple_projection(
        values_binary, values_rank, values_direct, coords, omega, bias, scale
    )
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[2], expected[2], rtol=1e-12, atol=1e-12)


def test_compute_sum_z_pyfunc_matches_explicit_matrix_sum() -> None:
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(20, 4))
    omega = rng.normal(size=(9, 4))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=9)
    scale = np.sqrt(2.0 / omega.shape[0])

    got = compute_sum_z.py_func(coords, omega, bias, scale)
    expected = (scale * np.cos(coords @ omega.T + bias)).sum(axis=0)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_compute_column_variances_handles_empty_and_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    omega = rng.normal(size=(6, 2))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=6)
    scale = np.sqrt(2.0 / omega.shape[0])

    empty = compute_column_variances(
        coords=np.empty((0, 2), dtype=np.float64),
        omega=omega,
        bias=bias,
        scale=scale,
    )
    np.testing.assert_array_equal(empty, np.zeros(6, dtype=np.float64))

    coords = rng.normal(size=(300, 2))
    v1 = compute_column_variances(coords, omega, bias, scale, max_samples=50, random_state=7)
    v2 = compute_column_variances(coords, omega, bias, scale, max_samples=50, random_state=7)
    np.testing.assert_allclose(v1, v2, rtol=0, atol=0)


def test_compute_cov_frobenius_per_scale_matches_manual_and_empty() -> None:
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(30, 2))
    omega = rng.normal(size=(10, 2))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=10)
    scale = np.sqrt(2.0 / omega.shape[0])
    offsets = np.array([0, 3, 7, 10], dtype=np.int64)

    got_frob, got_row4 = compute_cov_frobenius_per_scale(
        coords=coords,
        omega=omega,
        bias=bias,
        scale=scale,
        scale_offsets=offsets,
        max_samples=1000,
    )

    exp_frob = []
    exp_row4 = []
    for s in range(len(offsets) - 1):
        lo, hi = offsets[s], offsets[s + 1]
        Zs = scale * np.cos(coords @ omega[lo:hi].T + bias[lo:hi])
        Zs = Zs - Zs.mean(axis=0)
        gram = (Zs.T @ Zs) / coords.shape[0]
        exp_frob.append(np.sum(gram * gram))
        exp_row4.append(np.mean(np.sum(Zs ** 2, axis=1) ** 2))

    np.testing.assert_allclose(got_frob, np.array(exp_frob), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_row4, np.array(exp_row4), rtol=1e-12, atol=1e-12)

    empty_frob, empty_row4 = compute_cov_frobenius_per_scale(
        coords=np.empty((0, 2), dtype=np.float64),
        omega=omega,
        bias=bias,
        scale=scale,
        scale_offsets=offsets,
    )
    np.testing.assert_array_equal(empty_frob, np.zeros(len(offsets) - 1))
    np.testing.assert_array_equal(empty_row4, np.zeros(len(offsets) - 1))
