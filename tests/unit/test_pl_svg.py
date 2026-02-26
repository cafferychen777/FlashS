from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
plt = pytest.importorskip("matplotlib.pyplot")

from flashs.pl import _svg as pl_svg


def _make_adata_for_plotting() -> ad.AnnData:
    X = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 1.0],
            [4.0, 0.0, 2.0],
            [5.0, 1.0, 3.0],
        ]
    )
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["g1", "g2", "g3"]))
    adata.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]]
    )
    adata.var["flashs_pvalue"] = np.array([0.001, 0.2, 0.4])
    adata.var["flashs_qvalue"] = np.array([0.01, 0.25, 0.6])
    adata.var["flashs_effect_size"] = np.array([1.2, 0.4, 0.1])
    adata.uns["flashs"] = {"spatial_key": "spatial"}
    return adata


def test_pl_spatial_variable_genes_returns_figure_and_supports_save(tmp_path) -> None:
    adata = _make_adata_for_plotting()
    out = tmp_path / "svg_plot.png"

    fig = pl_svg.spatial_variable_genes(
        adata,
        key="flashs",
        n_top=2,
        ncols=2,
        save=str(out),
        show=False,
    )
    assert fig is not None
    assert out.exists()
    plt.close(fig)


def test_pl_spatial_variable_genes_validates_required_columns_and_spatial_key() -> None:
    adata = _make_adata_for_plotting()
    del adata.var["flashs_qvalue"]
    with pytest.raises(KeyError, match="flashs_qvalue"):
        pl_svg.spatial_variable_genes(adata, key="flashs", show=False)

    adata = _make_adata_for_plotting()
    adata.uns["flashs"]["spatial_key"] = "missing_spatial"
    with pytest.raises(KeyError, match="missing_spatial"):
        pl_svg.spatial_variable_genes(adata, key="flashs", show=False)


def test_pl_volcano_returns_figure_and_validates_columns(tmp_path) -> None:
    adata = _make_adata_for_plotting()
    out = tmp_path / "volcano.png"

    fig = pl_svg.volcano(adata, key="flashs", save=str(out), show=False)
    assert fig is not None
    assert out.exists()
    plt.close(fig)

    adata_bad = _make_adata_for_plotting()
    del adata_bad.var["flashs_effect_size"]
    with pytest.raises(KeyError, match="flashs_effect_size"):
        pl_svg.volcano(adata_bad, key="flashs", show=False)

