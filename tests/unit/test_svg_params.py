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


def test_bandwidths_property_requires_fit() -> None:
    model = FlashS()
    with pytest.raises(RuntimeError, match="Model not fitted yet"):
        _ = model.bandwidths


def test_fit_rejects_empty_coords() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        FlashS().fit(np.empty((0, 2), dtype=np.float64))


def test_fit_accepts_1d_coords_by_reshaping() -> None:
    coords_1d = np.linspace(0.0, 1.0, 20)
    model = FlashS(n_features=8, n_scales=2, random_state=0).fit(coords_1d)
    assert model._coords.shape == (20, 1)


def test_test_requires_fit_and_cell_count_match(small_coords, small_sparse_expr) -> None:
    model = FlashS(n_features=8, n_scales=2, random_state=0)
    with pytest.raises(RuntimeError, match="Must call fit"):
        model.test(small_sparse_expr)

    model.fit(small_coords)
    bad_x = small_sparse_expr[:-1, :]
    with pytest.raises(ValueError, match="does not match fitted coordinates"):
        model.test(bad_x)


def test_test_returns_empty_result_when_no_gene_passes_threshold() -> None:
    rng = np.random.default_rng(123)
    n_cells, n_genes = 120, 6
    coords = rng.normal(size=(n_cells, 2))

    # Each gene has only 2 expressing cells -> below asymptotic cutoff 30
    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    for g in range(n_genes):
        idx = rng.choice(n_cells, 2, replace=False)
        X[idx, g] = 1.0

    result = FlashS(n_features=8, n_scales=2, min_expressed=1, random_state=0).fit_test(
        coords,
        sparse.csr_matrix(X),
    )

    assert result.n_tested == 0
    assert result.tested_mask is not None
    assert not result.tested_mask.any()
    assert np.all(result.pvalues == 1.0)
    assert np.all(result.qvalues == 1.0)
