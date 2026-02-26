from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from flashs.preprocessing.normalize import log1p_transform, normalize_total


def test_normalize_total_dense_and_sparse_row_sums() -> None:
    dense = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    sparse_x = sparse.csr_matrix(dense)

    target = 100.0
    dense_norm = normalize_total(dense, target_sum=target)
    sparse_norm = normalize_total(sparse_x, target_sum=target).toarray()

    assert np.isclose(dense_norm[1].sum(), target)
    assert np.isclose(sparse_norm[1].sum(), target)
    assert np.allclose(dense_norm[0], 0.0)
    assert np.allclose(sparse_norm[0], 0.0)


def test_normalize_total_all_zero_dense_has_no_nan() -> None:
    dense = np.zeros((3, 4), dtype=np.float64)
    out = normalize_total(dense, target_sum=None)
    assert not np.isnan(out).any()
    assert np.allclose(out, 0.0)


def test_normalize_total_dense_inplace_modifies_input() -> None:
    dense = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    out = normalize_total(dense, target_sum=10.0, inplace=True)
    assert out is dense
    np.testing.assert_allclose(dense.sum(axis=1), np.array([10.0, 10.0]))


def test_log1p_transform_dense_and_sparse_with_base() -> None:
    dense = np.array([[0.0, 3.0], [7.0, 0.0]], dtype=np.float64)
    sparse_x = sparse.csr_matrix(dense)

    base = 2.0
    got_dense = log1p_transform(dense, base=base)
    got_sparse = log1p_transform(sparse_x, base=base).toarray()
    expected = np.log1p(dense) / np.log(base)

    np.testing.assert_allclose(got_dense, expected)
    np.testing.assert_allclose(got_sparse, expected)
