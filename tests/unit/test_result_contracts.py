from __future__ import annotations

import builtins

import numpy as np
import pytest

import flashs
from flashs.io import run_flashs as io_run_flashs
from flashs.model.svg import FlashSResult


def _make_result(with_projections: bool = True) -> FlashSResult:
    projections = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]) if with_projections else None
    return FlashSResult(
        gene_names=["g1", "g2", "g3"],
        pvalues=np.array([0.2, 0.01, 0.5]),
        qvalues=np.array([0.2, 0.03, 0.6]),
        statistics=np.array([1.0, 3.0, 0.5]),
        effect_size=np.array([0.1, 0.8, 0.05]),
        n_tested=3,
        pvalues_binary=np.array([0.3, 0.04, 0.7]),
        pvalues_rank=np.array([0.25, 0.05, 0.8]),
        n_expressed=np.array([10, 15, 3]),
        tested_mask=np.array([True, True, False]),
        projections=projections,
    )


def test_significant_genes_and_dataframe_sorting() -> None:
    result = _make_result()
    assert result.significant_genes() == ["g2"]
    assert result.significant_genes(effect_threshold=0.9) == []

    df = result.to_dataframe()
    assert list(df.columns) == [
        "gene",
        "pvalue",
        "qvalue",
        "statistic",
        "effect_size",
        "pvalue_binary",
        "pvalue_rank",
        "n_expressed",
    ]
    assert list(df["gene"]) == ["g2", "g1", "g3"]


def test_to_dataframe_raises_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _make_result()
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "pandas":
            raise ImportError("mocked missing pandas")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pandas required"):
        result.to_dataframe()


def test_spatial_embedding_behaviors() -> None:
    result = _make_result(with_projections=True)
    emb = result.get_spatial_embedding()
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms[:2], np.ones(2), atol=1e-12)
    # zero vector stays zero after safe normalization
    assert norms[2] == 0.0

    subset = result.get_spatial_embedding(genes=["g2", "not_found"], normalize=False)
    np.testing.assert_allclose(subset, np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="Projections not available"):
        _make_result(with_projections=False).get_spatial_embedding()


def test_flashs_lazy_run_flashs_export_and_missing_attr() -> None:
    assert flashs.run_flashs is io_run_flashs
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(flashs, "not_a_real_attr")
