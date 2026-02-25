"""
AnnData integration for Flash-S.

Provides seamless integration with Scanpy/Squidpy ecosystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.result import SpatialTestResult
from ..model.svg import FlashS, FlashSResult

if TYPE_CHECKING:
    import anndata as ad


def _store_result(
    adata: "ad.AnnData",
    result: SpatialTestResult,
    key_added: str,
    extra_fields: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Store spatial test result in AnnData.var.

    Unified storage function for all spatial test results.

    Parameters
    ----------
    adata : AnnData
        Target AnnData object.
    result : SpatialTestResult
        Test result to store.
    key_added : str
        Key prefix for columns.
    extra_fields : dict, optional
        Additional fields to store: {column_name: values_array}.
    metadata : dict, optional
        Metadata to store in adata.uns[key_added].
    """
    # Initialize base columns with NaN
    adata.var[f"{key_added}_pvalue"] = np.nan
    adata.var[f"{key_added}_qvalue"] = np.nan
    adata.var[f"{key_added}_statistic"] = np.nan
    adata.var[f"{key_added}_effect_size"] = np.nan

    extra_items: list[tuple[str, np.ndarray]] = []
    if extra_fields:
        expected_len = len(result.gene_names)
        for col_name, values in extra_fields.items():
            arr = np.asarray(values)
            if arr.ndim != 1:
                raise ValueError(
                    f"extra_fields['{col_name}'] must be 1D, got shape {arr.shape}"
                )
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"extra_fields['{col_name}'] length ({arr.shape[0]}) "
                    f"does not match result length ({expected_len})"
                )
            extra_items.append((col_name, arr))
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
                default = 0
            else:
                default = np.nan
            adata.var[f"{key_added}_{col_name}"] = default

    # Vectorized assignment: find genes that exist in adata
    gene_mask = np.isin(result.gene_names, adata.var_names)
    valid_genes = [g for g, m in zip(result.gene_names, gene_mask) if m]
    valid_indices = np.where(gene_mask)[0]

    if len(valid_genes) > 0:
        # Use .loc with array indexing for vectorized assignment
        adata.var.loc[valid_genes, f"{key_added}_pvalue"] = result.pvalues[valid_indices]
        adata.var.loc[valid_genes, f"{key_added}_qvalue"] = result.qvalues[valid_indices]
        adata.var.loc[valid_genes, f"{key_added}_statistic"] = result.statistics[valid_indices]
        adata.var.loc[valid_genes, f"{key_added}_effect_size"] = result.effect_size[valid_indices]

        if extra_items:
            for col_name, values in extra_items:
                adata.var.loc[valid_genes, f"{key_added}_{col_name}"] = values[valid_indices]

    # Store metadata
    adata.uns[key_added] = {
        "n_tested": result.n_tested,
        "n_significant": result.n_significant,
        **(metadata or {}),
    }


def run_flashs(
    adata: "ad.AnnData",
    spatial_key: str = "spatial",
    layer: str | None = None,
    genes: list[str] | None = None,
    n_features: int = 500,
    n_scales: int = 7,
    min_expressed: int = 5,
    copy: bool = False,
    key_added: str = "flashs",
    random_state: int | None = 0,
) -> FlashSResult | "ad.AnnData":
    """
    Run Flash-S spatial test on AnnData object.

    Detects spatially variable genes using Flash-S algorithm.
    Results are stored in adata.var and optionally returned.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with spatial coordinates.
    spatial_key : str, default='spatial'
        Key in adata.obsm containing spatial coordinates.
    layer : str, optional
        Layer to use for expression. If None, uses adata.X.
    genes : list of str, optional
        Subset of genes to test. If None, tests all genes.
    n_features : int, default=500
        Number of random Fourier features.
    n_scales : int, default=7
        Number of bandwidth scales.
    min_expressed : int, default=5
        Minimum expressing cells.
    copy : bool, default=False
        Whether to return a copy of adata.
    key_added : str, default='flashs'
        Key prefix for storing results in adata.var.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : FlashSResult or AnnData
        FlashSResult if copy=False (results also stored in adata.var).
        Modified AnnData copy if copy=True.

    Examples
    --------
    >>> import scanpy as sc
    >>> from flashs.io import run_flashs
    >>>
    >>> adata = sc.read_h5ad("spatial_data.h5ad")
    >>> result = run_flashs(adata, spatial_key="spatial")
    >>>
    >>> # Access results
    >>> sig_genes = adata.var[adata.var[f"{key_added}_qvalue"] < 0.05].index
    """
    if copy:
        adata = adata.copy()

    # Extract data
    coords, X, gene_names = _extract_adata(adata, spatial_key, layer, genes)

    # Configure and run test
    model = FlashS(
        n_features=n_features,
        n_scales=n_scales,
        min_expressed=min_expressed,
        random_state=random_state,
    )
    result = model.fit_test(coords, X, gene_names)

    # Store results using unified function
    _store_result(
        adata=adata,
        result=result,
        key_added=key_added,
        extra_fields={
            "pvalue_binary": result.pvalues_binary,
            "pvalue_rank": result.pvalues_rank,
            "n_expressed": result.n_expressed,
        },
        metadata={
            "n_features": model.n_features,
            "n_scales": len(model.bandwidths),
            "bandwidths": model.bandwidths,
        },
    )

    return adata if copy else result


def _extract_adata(
    adata: "ad.AnnData",
    spatial_key: str,
    layer: str | None,
    genes: list[str] | None,
):
    """
    Extract spatial coordinates, expression, and gene names from AnnData.

    Internal helper to avoid code duplication in run_flashs.
    """
    # Validate spatial key
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates not found at adata.obsm['{spatial_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key])

    # Extract expression matrix
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X

    # Get gene names
    gene_names = list(adata.var_names)

    # Filter genes if specified
    if genes is not None:
        if not adata.var_names.is_unique:
            raise ValueError(
                "adata.var_names must be unique when selecting genes by name; "
                "call adata.var_names_make_unique() first"
            )

        name_to_idx = {name: i for i, name in enumerate(gene_names)}
        seen: set[str] = set()
        selected_indices: list[int] = []
        selected_names: list[str] = []
        for gene in genes:
            # Keep user order; skip duplicate requests to avoid redundant columns.
            if gene in seen:
                continue
            seen.add(gene)
            idx = name_to_idx.get(gene)
            if idx is not None:
                selected_indices.append(idx)
                selected_names.append(gene)

        if not selected_indices:
            raise ValueError("None of the specified genes found in adata")
        gene_indices = np.asarray(selected_indices, dtype=np.intp)
        X = X[:, gene_indices]
        gene_names = selected_names

    return coords, X, gene_names
