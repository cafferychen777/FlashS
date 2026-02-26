"""Plotting functions for spatial variable gene results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    from matplotlib.figure import Figure


def _get_spatial_coords(adata: "ad.AnnData", key: str) -> np.ndarray:
    """Resolve spatial coordinates from adata.uns metadata or common keys."""
    spatial_key = adata.uns.get(key, {}).get("spatial_key", "spatial")
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key])
    raise KeyError(
        f"Spatial coordinates not found at adata.obsm['{spatial_key}']. "
        f"Set spatial_key when calling flashs.tl.spatial_variable_genes()."
    )


def _require_columns(adata: "ad.AnnData", key: str, suffixes: list[str]) -> None:
    """Raise KeyError if any expected result column is missing."""
    for suffix in suffixes:
        col = f"{key}_{suffix}"
        if col not in adata.var.columns:
            raise KeyError(
                f"'{col}' not found in adata.var. "
                f"Run flashs.tl.spatial_variable_genes(adata) first."
            )


def _finalize_figure(
    fig: "Figure",
    save: str | None,
    show: bool,
) -> "Figure | None":
    """Save and/or show figure; return Figure if show=False."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def spatial_variable_genes(
    adata: "ad.AnnData",
    key: str = "flashs",
    n_top: int = 6,
    spot_size: float | None = None,
    ncols: int = 3,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot top spatially variable genes on spatial coordinates.

    Displays expression of the top SVGs ranked by q-value overlaid
    on spatial coordinates.

    Parameters
    ----------
    adata
        Annotated data object with Flash-S results in ``adata.var``.
    key
        Key prefix used in ``flashs.tl.spatial_variable_genes``.
    n_top
        Number of top genes to plot.
    spot_size
        Size of scatter points. ``None`` auto-detects.
    ncols
        Number of columns in subplot grid.
    cmap
        Colormap for expression values.
    figsize
        Figure size ``(width, height)``. ``None`` auto-computes.
    save
        Path to save figure. ``None`` does not save.
    show
        Whether to show figure with ``plt.show()``.

    Returns
    -------
    ``Figure`` if ``show=False``, otherwise ``None``.
    """
    import matplotlib.pyplot as plt

    _require_columns(adata, key, ["qvalue"])

    ranked = adata.var[f"{key}_qvalue"].dropna().sort_values()
    top_genes = list(ranked.index[:n_top])
    if not top_genes:
        raise ValueError("No genes with valid q-values found.")

    coords = _get_spatial_coords(adata, key)
    name_to_idx = {name: i for i, name in enumerate(adata.var_names)}

    n_genes = len(top_genes)
    nrows = int(np.ceil(n_genes / ncols))
    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    if spot_size is None:
        extent = np.ptp(coords, axis=0).max()
        spot_size = max(0.5, (extent / np.sqrt(adata.n_obs)) * 2)

    qval_col = f"{key}_qvalue"
    for i, gene in enumerate(top_genes):
        ax = axes[i // ncols, i % ncols]
        gene_idx = name_to_idx[gene]

        expr = adata.X[:, gene_idx]
        if hasattr(expr, "toarray"):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()

        qval = adata.var.loc[gene, qval_col]
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=expr,
            s=spot_size,
            cmap=cmap,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{gene} (q={qval:.2e})", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)

    for j in range(n_genes, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    return _finalize_figure(fig, save, show)


def volcano(
    adata: "ad.AnnData",
    key: str = "flashs",
    q_threshold: float = 0.05,
    effect_threshold: float = 0.0,
    n_label: int = 10,
    figsize: tuple[float, float] = (5, 4),
    save: str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Volcano plot of spatial variable gene results.

    Plots -log10(p-value) vs effect size for all tested genes.

    Parameters
    ----------
    adata
        Annotated data object with Flash-S results.
    key
        Key prefix used in ``flashs.tl.spatial_variable_genes``.
    q_threshold
        Q-value threshold for significance coloring.
    effect_threshold
        Effect size threshold for significance coloring.
    n_label
        Number of top genes to label.
    figsize
        Figure size.
    save
        Path to save figure.
    show
        Whether to show figure.

    Returns
    -------
    ``Figure`` if ``show=False``, otherwise ``None``.
    """
    import matplotlib.pyplot as plt

    _require_columns(adata, key, ["pvalue", "qvalue", "effect_size"])

    var = adata.var.dropna(subset=[f"{key}_pvalue"])
    pvals = var[f"{key}_pvalue"].values
    qvals = var[f"{key}_qvalue"].values
    effect = var[f"{key}_effect_size"].values
    neg_log_p = -np.log10(np.clip(pvals, 1e-300, 1))

    sig = (qvals < q_threshold) & (effect > effect_threshold)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(
        effect[~sig],
        neg_log_p[~sig],
        c="#AAAAAA",
        s=8,
        alpha=0.5,
        edgecolors="none",
        rasterized=True,
        label="NS",
    )
    ax.scatter(
        effect[sig],
        neg_log_p[sig],
        c="#E64B35",
        s=10,
        alpha=0.7,
        edgecolors="none",
        rasterized=True,
        label=f"q < {q_threshold}",
    )

    top_idx = np.argsort(pvals)[:n_label]
    for idx in top_idx:
        ax.annotate(
            var.index[idx],
            (effect[idx], neg_log_p[idx]),
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.axhline(-np.log10(q_threshold), ls="--", c="#999999", lw=0.8)
    ax.set_xlabel("Effect size")
    ax.set_ylabel("$-\\log_{10}$(p-value)")
    ax.legend(frameon=False, fontsize=8)

    return _finalize_figure(fig, save, show)
