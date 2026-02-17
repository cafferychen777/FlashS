from __future__ import annotations

import numpy as np
from scipy import sparse

from flashs import FlashS


def _null_dataset(seed: int = 123) -> tuple[np.ndarray, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    coords = rng.random((200, 2)) * 100
    expr = sparse.random(
        200,
        40,
        density=0.25,
        format="csr",
        random_state=rng,
        data_rvs=lambda n: (rng.poisson(2, size=n) + 1).astype(np.float64),
    )
    return coords, expr


def test_null_small_calibration_is_reasonable() -> None:
    coords, expr = _null_dataset()
    result = FlashS(n_features=24, n_scales=4, min_expressed=1, random_state=11).fit_test(coords, expr)

    sig_rate = float(np.mean(result.pvalues < 0.05)) if result.n_tested > 0 else 0.0
    assert sig_rate < 0.2


def test_injected_pattern_is_detectable() -> None:
    rng = np.random.default_rng(7)
    n_cells, n_genes = 220, 30
    coords = rng.random((n_cells, 2)) * 100
    expr = sparse.random(
        n_cells,
        n_genes,
        density=0.05,
        format="csr",
        random_state=rng,
        data_rvs=lambda n: (rng.poisson(1, size=n) + 1).astype(np.float64),
    ).toarray()

    # Inject a broad spatial pattern so nnz is safely above the small-sample cutoff.
    hotspot = coords[:, 0] > 60.0
    expr[hotspot, 0] = rng.poisson(8, size=np.sum(hotspot)) + 5

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    gene_names[0] = "Hotspot"

    result = FlashS(n_features=28, n_scales=4, min_expressed=1, random_state=9).fit_test(
        coords,
        expr,
        gene_names=gene_names,
    )

    idx = result.gene_names.index("Hotspot")
    assert result.pvalues[idx] < 0.2
