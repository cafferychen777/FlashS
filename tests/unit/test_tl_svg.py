"""Tests for flashs.tl.spatial_variable_genes (scanpy-style API)."""

from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

from flashs.model.svg import FlashSResult


def _make_result(gene_names: list[str]) -> FlashSResult:
    n = len(gene_names)
    pvalues = np.linspace(0.01, 0.2, n)
    return FlashSResult(
        gene_names=gene_names,
        pvalues=pvalues,
        qvalues=np.minimum(pvalues * 1.5, 1.0),
        statistics=np.arange(1, n + 1, dtype=float),
        effect_size=np.linspace(0.1, 0.9, n),
        n_tested=n,
        pvalues_binary=np.minimum(pvalues + 0.05, 1.0),
        pvalues_rank=np.minimum(pvalues + 0.1, 1.0),
        n_expressed=np.arange(5, 5 + n),
        tested_mask=np.ones(n, dtype=bool),
    )


def _make_adata() -> ad.AnnData:
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["g1", "g2", "g3"]))
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
    return adata


def test_tl_svg_inplace_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scanpy convention: inplace returns None, results stored in adata."""
    from flashs.tl import _svg

    class DummyFlashS:
        def __init__(self, **kwargs):
            self.n_features = kwargs.get("n_features", 500)
            self.bandwidths = [0.5, 1.0]

        def fit_test(self, coords, X, gene_names):
            return _make_result(gene_names)

    monkeypatch.setattr(_svg, "FlashS", DummyFlashS)

    adata = _make_adata()
    result = _svg.spatial_variable_genes(adata, copy=False)
    assert result is None
    assert "flashs_pvalue" in adata.var.columns
    assert "flashs_qvalue" in adata.var.columns
    assert "flashs" in adata.uns


def test_tl_svg_copy_returns_adata(monkeypatch: pytest.MonkeyPatch) -> None:
    """copy=True returns modified AnnData, original unchanged."""
    from flashs.tl import _svg

    class DummyFlashS:
        def __init__(self, **kwargs):
            self.n_features = kwargs.get("n_features", 500)
            self.bandwidths = [0.5, 1.0]

        def fit_test(self, coords, X, gene_names):
            return _make_result(gene_names)

    monkeypatch.setattr(_svg, "FlashS", DummyFlashS)

    adata = _make_adata()
    out = _svg.spatial_variable_genes(adata, copy=True)
    assert out is not adata
    assert "flashs_pvalue" in out.var.columns
    assert "flashs_pvalue" not in adata.var.columns


def test_tl_module_alias() -> None:
    """flashs.tl.svg is an alias for spatial_variable_genes."""
    import flashs

    assert flashs.tl.svg is flashs.tl.spatial_variable_genes


def test_submodule_importable() -> None:
    """tl is importable as flashs.tl."""
    import flashs

    assert hasattr(flashs, "tl")
    assert hasattr(flashs.tl, "spatial_variable_genes")
