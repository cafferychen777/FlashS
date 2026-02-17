from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from flashs.preprocessing.normalize import normalize_total


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
