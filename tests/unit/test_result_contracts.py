from __future__ import annotations

import builtins

import numpy as np
import pytest

import flashs
from flashs.io import run_flashs as io_run_flashs
from flashs.model.svg import FlashSResult


def _make_result(with_projections: bool = True) -> FlashSResult:
    # g1 and g2 are tested, g3 is not (too few cells).
    # projections is compact (n_tested=2, D=2).
    projections = np.array([[3.0, 4.0], [1.0, 0.0]]) if with_projections else None
    return FlashSResult(
        gene_names=["g1", "g2", "g3"],
        pvalues=np.array([0.2, 0.01, 1.0]),
        qvalues=np.array([0.2, 0.03, 1.0]),
        statistics=np.array([1.0, 3.0, 0.0]),
        effect_size=np.array([0.1, 0.8, 0.0]),
        n_tested=2,
        pvalues_binary=np.array([0.3, 0.04, 1.0]),
        pvalues_rank=np.array([0.25, 0.05, 1.0]),
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
    # Compact projections: 2 rows for g1 and g2
    emb = result.get_spatial_embedding()
    assert emb.shape == (2, 2)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-12)

    # Select by name: g2 is tested, not_found is skipped
    subset = result.get_spatial_embedding(genes=["g2", "not_found"], normalize=False)
    np.testing.assert_allclose(subset, np.array([[1.0, 0.0]]))

    # g3 is untested — excluded from projections
    subset_g3 = result.get_spatial_embedding(genes=["g3"], normalize=False)
    assert subset_g3.shape == (0, 2)

    with pytest.raises(ValueError, match="Projections not available"):
        _make_result(with_projections=False).get_spatial_embedding()


def test_flashs_lazy_run_flashs_export_and_missing_attr() -> None:
    assert flashs.run_flashs is io_run_flashs
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(flashs, "not_a_real_attr")


def test_result_invariants_reject_inconsistent_tested_mask_length() -> None:
    with pytest.raises(ValueError, match="tested_mask length"):
        FlashSResult(
            gene_names=["g1", "g2", "g3"],
            pvalues=np.array([0.2, 0.01, 1.0]),
            qvalues=np.array([0.2, 0.03, 1.0]),
            statistics=np.array([1.0, 3.0, 0.0]),
            effect_size=np.array([0.1, 0.8, 0.0]),
            n_tested=2,
            pvalues_binary=np.array([0.3, 0.04, 1.0]),
            pvalues_rank=np.array([0.25, 0.05, 1.0]),
            n_expressed=np.array([10, 15, 3]),
            tested_mask=np.array([True, True]),
            projections=np.array([[3.0, 4.0], [1.0, 0.0]]),
        )


def test_result_invariants_reject_mismatched_n_tested_and_projection_rows() -> None:
    with pytest.raises(ValueError, match="tested_mask has 2 True entries"):
        FlashSResult(
            gene_names=["g1", "g2", "g3"],
            pvalues=np.array([0.2, 0.01, 1.0]),
            qvalues=np.array([0.2, 0.03, 1.0]),
            statistics=np.array([1.0, 3.0, 0.0]),
            effect_size=np.array([0.1, 0.8, 0.0]),
            n_tested=3,
            pvalues_binary=np.array([0.3, 0.04, 1.0]),
            pvalues_rank=np.array([0.25, 0.05, 1.0]),
            n_expressed=np.array([10, 15, 3]),
            tested_mask=np.array([True, True, False]),
            projections=np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]),
        )

    with pytest.raises(ValueError, match="projections has 1 rows"):
        FlashSResult(
            gene_names=["g1", "g2", "g3"],
            pvalues=np.array([0.2, 0.01, 1.0]),
            qvalues=np.array([0.2, 0.03, 1.0]),
            statistics=np.array([1.0, 3.0, 0.0]),
            effect_size=np.array([0.1, 0.8, 0.0]),
            n_tested=2,
            pvalues_binary=np.array([0.3, 0.04, 1.0]),
            pvalues_rank=np.array([0.25, 0.05, 1.0]),
            n_expressed=np.array([10, 15, 3]),
            tested_mask=np.array([True, True, False]),
            projections=np.array([[3.0, 4.0]]),
        )
