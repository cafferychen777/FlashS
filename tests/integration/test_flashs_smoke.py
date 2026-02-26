from __future__ import annotations

import numpy as np
from scipy import sparse

from flashs import FlashS


def test_flashs_fit_test_smoke_sparse(small_coords, small_sparse_expr, gene_names) -> None:
    model = FlashS(n_features=16, n_scales=4, min_expressed=1, random_state=0)
    result = model.fit_test(small_coords, small_sparse_expr, gene_names=gene_names)

    n_genes = small_sparse_expr.shape[1]
    assert result.n_tested > 0
    assert len(result.gene_names) == n_genes
    assert result.pvalues.shape == (n_genes,)
    assert result.tested_mask is not None
    assert result.tested_mask.sum() == result.n_tested
    assert np.all((result.pvalues > 0.0) & (result.pvalues <= 1.0))


def test_flashs_return_projections_shape(small_coords, small_sparse_expr, gene_names) -> None:
    model = FlashS(n_features=12, n_scales=3, min_expressed=1, random_state=0)
    result = model.fit(small_coords).test(
        small_sparse_expr,
        gene_names=gene_names,
        return_projections=True,
    )

    assert result.projections is not None
    assert result.projections.shape == (result.n_tested, model._rff.n_features)


def test_result_aligned_with_input_gene_order() -> None:
    """Results must align with input gene order, even when some genes are skipped."""
    rng = np.random.default_rng(99)
    n_cells, n_genes = 100, 20
    coords = rng.random((n_cells, 2)) * 50

    # Create expression: first 5 genes are near-empty (will be skipped)
    expr = np.zeros((n_cells, n_genes))
    for g in range(5):
        # Only 2 expressing cells — below min_expressed=5
        idx = rng.choice(n_cells, 2, replace=False)
        expr[idx, g] = rng.poisson(3, size=2) + 1
    for g in range(5, n_genes):
        idx = rng.choice(n_cells, 40, replace=False)
        expr[idx, g] = rng.poisson(5, size=40) + 1

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    expr_sp = sparse.csr_matrix(expr)

    result = FlashS(n_features=16, n_scales=3, min_expressed=5, random_state=0).fit_test(
        coords, expr_sp, gene_names=gene_names,
    )

    # All arrays must have n_genes length
    assert len(result.gene_names) == n_genes
    assert result.pvalues.shape == (n_genes,)
    assert result.qvalues.shape == (n_genes,)
    assert result.n_expressed.shape == (n_genes,)
    assert result.tested_mask.shape == (n_genes,)

    # Gene names must be in input order
    assert result.gene_names == gene_names

    # Skipped genes have conservative defaults
    for g in range(5):
        assert result.tested_mask[g] == False
        assert result.pvalues[g] == 1.0
        assert result.qvalues[g] == 1.0
        assert result.statistics[g] == 0.0
        # n_expressed is still populated
        assert result.n_expressed[g] >= 0

    # Tested genes have real values
    for g in range(5, n_genes):
        assert result.tested_mask[g] == True
        assert result.n_expressed[g] > 0

    assert result.n_tested == n_genes - 5
