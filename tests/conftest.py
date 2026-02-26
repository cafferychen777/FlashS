from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_coords(rng: np.random.Generator) -> np.ndarray:
    return rng.random((200, 2)) * 10.0


@pytest.fixture
def small_sparse_expr(rng: np.random.Generator) -> sparse.csr_matrix:
    data = sparse.random(
        200,
        12,
        density=0.25,
        format="csr",
        random_state=rng,
        data_rvs=lambda n: (rng.poisson(2, size=n) + 1).astype(np.float64),
    )
    return data


@pytest.fixture
def small_dense_expr(small_sparse_expr: sparse.csr_matrix) -> np.ndarray:
    return small_sparse_expr.toarray()


@pytest.fixture
def gene_names() -> list[str]:
    return [f"Gene_{i}" for i in range(12)]
