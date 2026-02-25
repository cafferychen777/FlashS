from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

from flashs.io.anndata import _extract_adata, _store_result, run_flashs
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
    adata.layers["counts"] = X + 10.0
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
    return adata


def test_extract_adata_reads_layer_and_filters_genes_in_var_order() -> None:
    adata = _make_adata()
    coords, X, gene_names = _extract_adata(
        adata=adata,
        spatial_key="spatial",
        layer="counts",
        genes=["g3", "g1"],
    )

    np.testing.assert_allclose(coords, adata.obsm["spatial"])
    np.testing.assert_allclose(X, adata.layers["counts"][:, [0, 2]])
    # _extract_adata keeps AnnData var order, not user-provided order.
    assert gene_names == ["g1", "g3"]


def test_extract_adata_validates_required_inputs() -> None:
    adata = _make_adata()

    with pytest.raises(KeyError, match="Spatial coordinates not found"):
        _extract_adata(adata=adata, spatial_key="missing", layer=None, genes=None)

    with pytest.raises(KeyError, match="Layer 'missing' not found"):
        _extract_adata(adata=adata, spatial_key="spatial", layer="missing", genes=None)

    with pytest.raises(ValueError, match="None of the specified genes found"):
        _extract_adata(
            adata=adata,
            spatial_key="spatial",
            layer=None,
            genes=["not_in_adata"],
        )


def test_store_result_writes_matching_genes_and_metadata() -> None:
    adata = _make_adata()
    result = _make_result(["g2", "missing_gene", "g3"])

    _store_result(
        adata=adata,
        result=result,
        key_added="flashs",
        extra_fields={
            "pvalue_binary": result.pvalues_binary,
            "n_expressed": result.n_expressed,
        },
        metadata={"n_scales": 2},
    )

    # Matched genes are assigned from result arrays.
    assert adata.var.loc["g2", "flashs_pvalue"] == pytest.approx(result.pvalues[0])
    assert adata.var.loc["g3", "flashs_qvalue"] == pytest.approx(result.qvalues[2])
    # Unmatched genes keep defaults.
    assert np.isnan(adata.var.loc["g1", "flashs_pvalue"])
    assert adata.var.loc["g1", "flashs_n_expressed"] == 0

    assert adata.uns["flashs"]["n_tested"] == result.n_tested
    assert adata.uns["flashs"]["n_significant"] == result.n_significant
    assert adata.uns["flashs"]["n_scales"] == 2


def test_run_flashs_copy_modes_with_stubbed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adata = _make_adata()

    class DummyFlashS:
        def __init__(
            self,
            n_features: int,
            n_scales: int,
            min_expressed: int,
            random_state: int | None,
        ) -> None:
            self.n_features = n_features
            self.n_scales = n_scales
            self.min_expressed = min_expressed
            self.random_state = random_state
            self.bandwidths = [0.25, 0.5, 1.0]

        def fit_test(
            self,
            coords: np.ndarray,
            X: np.ndarray,
            gene_names: list[str],
        ) -> FlashSResult:
            assert coords.shape[0] == X.shape[0]
            return _make_result(gene_names)

    monkeypatch.setattr("flashs.io.anndata.FlashS", DummyFlashS)

    result = run_flashs(adata, key_added="k", copy=False, n_features=16, n_scales=4)
    assert isinstance(result, FlashSResult)
    assert "k_pvalue" in adata.var.columns
    assert adata.uns["k"]["n_features"] == 16
    assert adata.uns["k"]["n_scales"] == 3  # length of DummyFlashS.bandwidths

    adata2 = _make_adata()
    out = run_flashs(adata2, key_added="k2", copy=True)
    assert out is not adata2
    assert "k2_pvalue" in out.var.columns
    assert "k2_pvalue" not in adata2.var.columns
