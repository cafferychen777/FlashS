"""
AnnData integration for Flash-S.

Provides:
- ``_extract_adata``: extract coords/expression/gene_names from AnnData
- ``_store_result``: write SpatialTestResult into adata.var / adata.uns
- ``run_flashs``: convenience wrapper (delegates to ``flashs.tl``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.result import SpatialTestResult

if TYPE_CHECKING:
    import anndata as ad

    from ..model.svg import FlashSResult


def _extract_adata(
    adata: "ad.AnnData",
    spatial_key: str,
    layer: str | None,
    genes: list[str] | None,
):
    """Extract spatial coordinates, expression, and gene names from AnnData."""
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates not found at adata.obsm['{spatial_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key])

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X

    gene_names = list(adata.var_names)

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


def _store_result(
    adata: "ad.AnnData",
    result: SpatialTestResult,
    key_added: str,
    extra_fields: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Store spatial test result in AnnData.var and adata.uns.

    Parameters
    ----------
    adata
        Target AnnData object.
    result
        Test result to store.
    key_added
        Key prefix for columns.
    extra_fields
        Additional fields: ``{column_name: values_array}``.
    metadata
        Metadata to store in ``adata.uns[key_added]``.
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

    # Positional assignment: map result genes to adata.var positions.
    # Uses iloc-style indexing to handle duplicate var_names correctly.
    var_name_list = list(adata.var_names)
    name_to_pos: dict[str, int] = {}
    for i, name in enumerate(var_name_list):
        if name not in name_to_pos:
            name_to_pos[name] = i

    var_positions: list[int] = []
    result_indices: list[int] = []
    for j, gene in enumerate(result.gene_names):
        pos = name_to_pos.get(gene)
        if pos is not None:
            var_positions.append(pos)
            result_indices.append(j)

    if var_positions:
        pos = np.array(var_positions)
        idx = np.array(result_indices)
        adata.var[f"{key_added}_pvalue"].values[pos] = result.pvalues[idx]
        adata.var[f"{key_added}_qvalue"].values[pos] = result.qvalues[idx]
        adata.var[f"{key_added}_statistic"].values[pos] = result.statistics[idx]
        adata.var[f"{key_added}_effect_size"].values[pos] = result.effect_size[idx]

        for col_name, values in extra_items:
            adata.var[f"{key_added}_{col_name}"].values[pos] = values[idx]

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
) -> "FlashSResult | ad.AnnData":
    """
    Run Flash-S spatial test on AnnData object.

    Convenience wrapper around :func:`flashs.tl.spatial_variable_genes`.
    Delegates all work to the scanpy-style API; exists for backward
    compatibility.

    Returns
    -------
    result : FlashSResult or AnnData
        ``FlashSResult`` if ``copy=False`` (results also stored in adata.var).
        Modified AnnData copy if ``copy=True``.
    """
    from ..tl._svg import _run_and_store

    if copy:
        adata = adata.copy()

    result = _run_and_store(
        adata, spatial_key, layer, genes,
        n_features, n_scales, min_expressed, key_added, random_state,
    )

    return adata if copy else result
