from __future__ import annotations

import pytest

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
