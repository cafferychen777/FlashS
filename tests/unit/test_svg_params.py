from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from flashs import FlashS


def test_constructor_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        FlashS(n_features=0)
    with pytest.raises(ValueError):
        FlashS(n_scales=0)
    with pytest.raises(ValueError):
        FlashS(min_expressed=-1)
    with pytest.raises(ValueError):
        FlashS(bandwidth=[])
    with pytest.raises(ValueError):
        FlashS(bandwidth=[-1.0, 2.0])


def test_bandwidth_is_normalized_to_list() -> None:
    scalar = FlashS(bandwidth=1.5)
    assert scalar.bandwidth == [1.5]

    array_like = FlashS(bandwidth=[0.5, 1.0, 2.0])
    assert array_like.bandwidth == [0.5, 1.0, 2.0]


def test_gene_names_length_validation(small_coords, small_sparse_expr) -> None:
    model = FlashS(n_features=12, n_scales=3, min_expressed=1, random_state=1)
    model.fit(small_coords)

    with pytest.raises(ValueError, match="gene_names length"):
        model.test(small_sparse_expr, gene_names=["only_one_name"])


def test_auto_normalize_uses_deterministic_global_max() -> None:
    model = FlashS()
    row = np.ones(10001, dtype=np.float64)
    row[-1] = 1000.0
    X = sparse.csr_matrix(row.reshape(1, -1))

    out = model._normalize_expression(
        X=X,
        normalize="auto",
        log_transform=False,
        verbose=False,
    )
    total = float(np.asarray(out.sum(axis=1)).ravel()[0])
    assert total == pytest.approx(1e4)
