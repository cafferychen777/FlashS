"""Spatial variable gene detection (scanpy-style API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

from ..io.anndata import _extract_adata, _store_result
from ..model.svg import FlashS, FlashSResult


def _run_and_store(
    adata: "ad.AnnData",
    spatial_key: str,
    layer: str | None,
    genes: list[str] | None,
    n_features: int,
    n_scales: int,
    min_expressed: int,
    key_added: str,
    random_state: int | None,
) -> FlashSResult:
    """Run FlashS on *adata* and store results in adata.var / adata.uns.

    Single source of truth for both ``flashs.tl.spatial_variable_genes``
    and ``flashs.io.run_flashs``. Returns the full FlashSResult so callers
    that need it (backward-compat ``run_flashs``) don't have to reconstruct.
    """
    coords, X, gene_names, var_indices = _extract_adata(
        adata, spatial_key, layer, genes,
    )

    model = FlashS(
        n_features=n_features,
        n_scales=n_scales,
        min_expressed=min_expressed,
        random_state=random_state,
    )
    result = model.fit_test(coords, X, gene_names)

    _store_result(
        adata=adata,
        result=result,
        var_indices=var_indices,
        key_added=key_added,
        extra_fields={
            "pvalue_binary": result.pvalues_binary,
            "pvalue_rank": result.pvalues_rank,
            "n_expressed": result.n_expressed,
        },
        metadata={
            "spatial_key": spatial_key,
            "n_features": model.n_features,
            "n_scales": len(model.bandwidths),
            "bandwidths": model.bandwidths,
        },
    )

    return result


def spatial_variable_genes(
    adata: "ad.AnnData",
    spatial_key: str = "spatial",
    layer: str | None = None,
    genes: list[str] | None = None,
    n_features: int = 500,
    n_scales: int = 7,
    min_expressed: int = 5,
    key_added: str = "flashs",
    copy: bool = False,
    random_state: int | None = 0,
) -> "ad.AnnData | None":
    """
    Detect spatially variable genes using Flash-S.

    Uses Random Fourier Features and multi-kernel Cauchy combination
    to identify genes with spatial expression patterns. Results are
    stored in ``adata.var`` and ``adata.uns``.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates in
        ``adata.obsm[spatial_key]``.
    spatial_key
        Key in ``adata.obsm`` for spatial coordinates.
    layer
        Expression layer to use. ``None`` means ``adata.X``.
    genes
        Subset of genes to test. ``None`` tests all genes.
    n_features
        Number of Random Fourier Features (D).
    n_scales
        Number of bandwidth scales (L).
    min_expressed
        Minimum number of expressing cells to test a gene.
    key_added
        Key prefix for results in ``adata.var`` and ``adata.uns``.
    copy
        Whether to return a modified copy instead of updating in place.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    Returns ``None`` if ``copy=False`` (results stored in ``adata``),
    otherwise a modified copy of ``adata``.

    The following fields are added:

    ``adata.var['{key_added}_pvalue']``
        Raw p-values (Cauchy combination across all kernels).
    ``adata.var['{key_added}_qvalue']``
        FDR-adjusted q-values (Benjamini-Hochberg).
    ``adata.var['{key_added}_statistic']``
        Combined test statistics.
    ``adata.var['{key_added}_effect_size']``
        Spatial effect size (variance explained by spatial kernels).
    ``adata.var['{key_added}_pvalue_binary']``
        P-values from binary (presence/absence) test.
    ``adata.var['{key_added}_pvalue_rank']``
        P-values from rank-transformed test.
    ``adata.var['{key_added}_n_expressed']``
        Number of expressing cells per gene.
    ``adata.uns['{key_added}']``
        Dictionary with metadata (n_tested, n_significant, bandwidths).

    Examples
    --------
    >>> import scanpy as sc
    >>> import flashs
    >>> adata = sc.read_h5ad("spatial.h5ad")
    >>> flashs.tl.spatial_variable_genes(adata)
    >>> sig = adata.var.query("flashs_qvalue < 0.05")
    """
    if copy:
        adata = adata.copy()

    _run_and_store(
        adata, spatial_key, layer, genes,
        n_features, n_scales, min_expressed, key_added, random_state,
    )

    return adata if copy else None
