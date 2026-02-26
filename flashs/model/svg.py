"""
Flash-S: Frequency-domain Linearized Adaptive Spatial Hypothesis testing via Sketching.

Ultra-fast spatial expression pattern detection for million-scale
spatial transcriptomics data using Random Fourier Features and
true sketching (no N×D matrix storage).

Complexity:
- fit():  O(N·D) centering + O(M·D) variances + O(M·D²/L) covariance norms
- test(): O(nnz·D + nnz·log nnz) per gene (sparse projection + rank transform)
- Total:  O(N·D + G·nnz_avg·D) dominated by test() for G >> 1
- Memory: O(D) per gene, O(M·D/L) peak during fit (streamed per-scale)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import rankdata

from ..core.bandwidth import compute_adaptive_scales
from ..core.pvalue import (
    adjust_pvalues,
    batch_cauchy_combination,
)
from ..core.result import SpatialTestResult
from ..core.rff import KernelType, RFFParams, sample_spectral_frequencies
from ..core.sketch import (
    _project_triple,
    _project_triple_2d,
    compute_column_variances,
    compute_cov_frobenius_per_scale,
    compute_sum_z,
)
from ..preprocessing.normalize import log1p_transform, normalize_total


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FlashSResult(SpatialTestResult):
    """
    Results from Flash-S spatial variable gene test.

    The primary p-values come from multi-kernel Cauchy combination across
    {binary, rank, direct} tests × L bandwidth scales + projection kernels.

    All arrays are aligned with the input gene order. Genes that were not
    tested (too few expressing cells) receive pvalues=1.0, statistics=0.0.
    Use ``tested_mask`` to distinguish tested from untested genes.

    Extends SpatialTestResult with SVG-specific fields:
    - Binary/rank p-values (diagnostics)
    - Expression counts per gene
    - Boolean mask of actually tested genes
    - RFF projection vectors for spatial clustering
    """

    pvalues_binary: NDArray[np.floating]
    """P-values from binary (presence/absence) pattern test (diagnostic)."""

    pvalues_rank: NDArray[np.floating]
    """P-values from rank (intensity) pattern test (diagnostic)."""

    n_expressed: NDArray[np.integer]
    """Number of expressing cells per gene."""

    tested_mask: NDArray[np.bool_] | None = None
    """Boolean mask: True for genes where the asymptotic test was performed.

    Genes are excluded (tested_mask=False, pvalue=1.0) when they have
    fewer than ``min_expressed`` expressing cells or fewer than 30 expressing
    cells (minimum for chi-square asymptotic validity).
    """

    projections: NDArray[np.floating] | None = None
    """RFF projection vectors (n_genes, D) for spatial clustering.

    Each row is a gene's "spatial frequency fingerprint". Genes with similar
    spatial patterns have high cosine similarity. Use get_spatial_embedding()
    for L2-normalized projections suitable for clustering.
    """

    def _build_dataframe_dict(self) -> dict:
        """Extend base dict with SVG-specific columns."""
        d = super()._build_dataframe_dict()
        d.update({
            "pvalue_binary": self.pvalues_binary,
            "pvalue_rank": self.pvalues_rank,
            "n_expressed": self.n_expressed,
        })
        return d

    def get_spatial_embedding(
        self,
        genes: list[str] | None = None,
        normalize: bool = True,
    ) -> NDArray[np.floating]:
        """
        Get spatial shape embedding for clustering genes.

        Returns L2-normalized RFF projections that capture spatial shape
        independent of expression level.

        Parameters
        ----------
        genes : list[str], optional
            Genes to include. If None, uses all tested genes.
        normalize : bool, default=True
            Whether to L2-normalize (recommended for clustering).

        Returns
        -------
        embedding : ndarray of shape (n_genes, D)
            Spatial shape embedding.
        """
        if self.projections is None:
            raise ValueError(
                "Projections not available. Run FlashS.test() with "
                "return_projections=True to enable spatial embedding."
            )

        if genes is None:
            proj = self.projections
        else:
            name_to_idx = {n: i for i, n in enumerate(self.gene_names)}
            indices = [name_to_idx[g] for g in genes if g in name_to_idx]
            proj = self.projections[indices]

        if normalize:
            norms = np.linalg.norm(proj, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            proj = proj / norms

        return proj


# ---------------------------------------------------------------------------
# Internal dataclasses for the sketch → infer pipeline
# ---------------------------------------------------------------------------

@dataclass
class _NullParams:
    """Null distribution parameters shared across all genes.

    Derived from fit()-phase RFF statistics. Used by _compute_kernel_pvalues
    to map test statistics → p-values via kurtosis-corrected Satterthwaite.
    """

    n_cells: int
    n_dims: int
    mean_T_rank: float       # n_cells * sum(z_variances)
    sum_cov_frob_sq: float   # sum of per-scale ||Cov_z||_F^2
    sum_row_norm4: float     # sum of per-scale E[||z_{c,i}||^4]
    ps_z_var: NDArray[np.float64]    # (n_scales,) per-scale sum of z variances
    ps_frob_sq: NDArray[np.float64]  # (n_scales,) per-scale Frobenius norms
    row_norm4: NDArray[np.float64]   # (n_scales,) per-scale kurtosis terms
    has_projection: bool


@dataclass
class _PerGeneStats:
    """Per-gene sketch outputs from a single gene."""

    T_binary: float
    T_rank: float
    scale_binary: float
    df_binary: float
    mean_T_binary: float
    ps_T_binary: NDArray[np.float64]
    ps_T_rank: NDArray[np.float64]
    ps_T_direct: NDArray[np.float64]
    y_var_binary: float
    kappa4_binary: float
    kappa4_rank: float
    kappa4_direct: float
    T_proj_binary: float
    T_proj_rank: float
    T_proj_direct: float
    rank_projection: NDArray[np.float64] | None


@dataclass
class _GeneStats:
    """Vectorized per-gene statistics for batch p-value computation.

    Each array has shape (n_tested,) or (n_tested, n_scales).
    """

    input_indices: list[int]
    T_binary: NDArray[np.float64]
    T_rank: NDArray[np.float64]
    scale_binary: NDArray[np.float64]
    df_binary: NDArray[np.float64]
    mean_T_binary: NDArray[np.float64]
    kappa4_binary: NDArray[np.float64]
    kappa4_rank: NDArray[np.float64]
    kappa4_direct: NDArray[np.float64]
    ps_T_binary: NDArray[np.float64]
    ps_T_rank: NDArray[np.float64]
    ps_T_direct: NDArray[np.float64]
    y_var_binary: NDArray[np.float64]
    T_proj_binary: NDArray[np.float64]
    T_proj_rank: NDArray[np.float64]
    T_proj_direct: NDArray[np.float64]

    @classmethod
    def from_list(
        cls,
        stats: list[_PerGeneStats],
        input_indices: list[int],
    ) -> _GeneStats:
        """Vectorize a list of per-gene stats into batch arrays."""
        return cls(
            input_indices=input_indices,
            T_binary=np.array([s.T_binary for s in stats]),
            T_rank=np.array([s.T_rank for s in stats]),
            scale_binary=np.array([s.scale_binary for s in stats]),
            df_binary=np.array([s.df_binary for s in stats]),
            mean_T_binary=np.array([s.mean_T_binary for s in stats]),
            kappa4_binary=np.array([s.kappa4_binary for s in stats]),
            kappa4_rank=np.array([s.kappa4_rank for s in stats]),
            kappa4_direct=np.array([s.kappa4_direct for s in stats]),
            ps_T_binary=np.array([s.ps_T_binary for s in stats]),
            ps_T_rank=np.array([s.ps_T_rank for s in stats]),
            ps_T_direct=np.array([s.ps_T_direct for s in stats]),
            y_var_binary=np.array([s.y_var_binary for s in stats]),
            T_proj_binary=np.array([s.T_proj_binary for s in stats]),
            T_proj_rank=np.array([s.T_proj_rank for s in stats]),
            T_proj_direct=np.array([s.T_proj_direct for s in stats]),
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------

def _excess_kurtosis_zi(
    values: NDArray[np.float64],
    mean: float,
    std: float,
    n_zero: int,
    n_cells: int,
) -> float:
    """Excess kurtosis of a zero-inflated vector (rank or direct).

    Handles the zero entries analytically: their standardized value is
    -mean/std, contributing n_zero * (mean/std)^4 to the raw 4th moment.
    """
    a = mean / std
    fourth = (n_zero * a ** 4 + np.sum(((values - mean) / std) ** 4)) / n_cells
    return fourth - 3.0


def _satterthwaite_params(
    mean_T: NDArray[np.floating] | float,
    var_T: NDArray[np.floating] | float,
    fallback_df: float = 1.0,
) -> tuple[NDArray[np.floating] | float, NDArray[np.floating] | float]:
    """Compute Satterthwaite-approximation scale and df from E[T] and Var[T].

    Returns (scale, df) where T/scale ~ chi2(df).
    When mean_T is near zero, returns safe fallback values.
    """
    mean_T = np.asarray(mean_T)
    var_T = np.asarray(var_T)
    var_T = np.maximum(var_T, 1e-20)
    ok = mean_T > 1e-10
    safe_mean = np.where(ok, mean_T, 1.0)
    scale = np.where(ok, var_T / (2 * safe_mean), 1.0)
    df = np.where(ok, 2 * safe_mean ** 2 / var_T, fallback_df)
    if scale.ndim == 0:
        return float(scale), float(df)
    return scale, df


def _safe_chi2_sf(
    T: NDArray[np.float64],
    scale: NDArray[np.float64] | float,
    df: NDArray[np.float64] | float,
    mean_threshold: NDArray[np.float64] | float | None = None,
) -> NDArray[np.float64]:
    """Chi-square survival function with safe guards for near-zero scale.

    When mean_threshold is provided, returns 1.0 (no evidence) for entries
    where mean_threshold <= 1e-10.
    """
    scale = np.asarray(scale)
    safe_scale = np.where(scale > 1e-10, scale, 1.0)
    pvals = _chi2_dist.sf(T / safe_scale, df)
    if mean_threshold is not None:
        pvals = np.where(np.asarray(mean_threshold) > 1e-10, pvals, 1.0)
    return pvals


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_MIN_CELLS_ASYMPTOTIC = 30
"""Minimum expressing cells for chi-square asymptotic validity (CLT threshold)."""


class FlashS:
    """
    Flash-S spatial expression pattern detector.

    Detects spatially variable genes using Random Fourier Features
    with true sketching — no N×D feature matrix is ever stored.

    Complexity
    ----------
    - fit():  O(N·D) centering + O(M·D) variances + O(M·D²/L) covariance norms
    - test(): O(nnz·D + nnz·log nnz) per gene (sparse projection + rank)
    - Total:  dominated by test() for G >> 1, since fit is amortized
    - Memory: O(D) per gene, O(M·D/L) peak during fit (streamed per-scale)

    The fit-phase O(N·D) is a single numba-parallel scan computing the
    exact centering vector sum_z. This cannot be subsampled because the
    estimation error enters T quadratically, creating O(N²/M) bias.

    Parameters
    ----------
    n_features : int, default=500
        Number of random Fourier features (D).
    n_scales : int, default=7
        Number of bandwidth scales for multi-scale analysis (L).
    kernel : KernelType, default=GAUSSIAN
        Kernel type for RFF.
    bandwidth : float or list[float], optional
        Manual bandwidth(s). If None, estimated adaptively.
    min_expressed : int, default=5
        Minimum expressing cells for a gene to be tested.
        The effective threshold is ``max(min_expressed, 30)`` because the
        chi-square asymptotic approximation requires at least 30 non-zero
        entries for CLT convergence.
    normalize : bool or "auto", default=False
        Whether to normalize expression data.
        - "auto": Normalize if max value > 100 (likely raw counts)
        - True: Always normalize (library size to 1e4)
        - False: Never normalize (recommended for raw counts)
    log_transform : bool, default=False
        Whether to apply log1p transformation.
    adjustment : str, default="bh"
        Multiple testing correction method.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from flashs import FlashS
    >>>
    >>> # Simulated data
    >>> n_cells, n_genes = 100000, 1000
    >>> coords = np.random.randn(n_cells, 2)
    >>> expr = sparse.random(n_cells, n_genes, density=0.01)
    >>>
    >>> # Run test
    >>> result = FlashS().fit(coords).test(expr)
    >>> print(f"Found {result.n_significant} significant genes")
    """

    def __init__(
        self,
        n_features: int = 500,
        n_scales: int = 7,
        kernel: KernelType = KernelType.GAUSSIAN,
        bandwidth: float | list[float] | None = None,
        min_expressed: int = 5,
        normalize: bool | str = False,
        log_transform: bool = False,
        adjustment: Literal["bh", "bonferroni", "holm", "by", "storey", "none"] = "bh",
        random_state: int | None = 0,
    ):
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {n_scales}")
        if min_expressed < 0:
            raise ValueError(f"min_expressed must be >= 0, got {min_expressed}")
        if bandwidth is not None:
            bandwidth = [float(bandwidth)] if isinstance(bandwidth, (int, float)) else [float(b) for b in bandwidth]
            if len(bandwidth) == 0:
                raise ValueError("bandwidth must be non-empty")
            if any(b <= 0 for b in bandwidth):
                raise ValueError(f"bandwidth values must be > 0, got {bandwidth}")

        self.n_features = n_features
        self.n_scales = n_scales
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.min_expressed = min_expressed
        self.normalize = normalize
        self.log_transform = log_transform
        self.adjustment = adjustment
        self.random_state = random_state

        self._fitted = False

        # Set by fit()
        self._rff: RFFParams
        self._bandwidths: list[float]
        self._scale_offsets: NDArray[np.intp]
        self._sum_z: NDArray[np.float64]
        self._z_variances: NDArray[np.float64]
        self._cov_frob_sq: NDArray[np.float64]
        self._row_norm4_per_scale: NDArray[np.float64]
        self._coords: NDArray[np.float64]

    @property
    def bandwidths(self) -> list[float]:
        """Bandwidth values used for multi-scale RFF."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._bandwidths

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, coords: NDArray[np.floating]) -> "FlashS":
        """
        Fit the model to spatial coordinates.

        Samples RFF parameters and precomputes null distribution statistics.

        Cost breakdown (M = min(N, 10000), D_s = D/L per scale):
          - sum_z (centering vector): O(N·D) exact, O(D) memory
          - column variances:         O(M·D) subsampled, O(D) memory
          - covariance norms:         O(M·D²/L) subsampled, O(M·D/L) memory
        The O(N·D) centering pass and O(M·D²/L) covariance pass are both
        negligible vs. test() which is O(G · nnz_avg · D) across all genes.

        Parameters
        ----------
        coords : ndarray of shape (n_cells, n_dims)
            Spatial coordinates (2D or 3D).

        Returns
        -------
        self : FlashS
            Fitted model.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        if coords.shape[0] == 0:
            raise ValueError("coords must have at least one cell (got 0 rows)")

        self._coords = coords
        rng = np.random.default_rng(self.random_state)

        n_dims = coords.shape[1]

        # Determine bandwidths (uses subsampling internally)
        if self.bandwidth is not None:
            bandwidths = self.bandwidth
        else:
            bandwidths = compute_adaptive_scales(
                coords,
                n_scales=self.n_scales,
                coverage="geometric",
                random_state=self.random_state,
            )

        # Distribute RFF features exactly across scales via divmod
        n_bw = len(bandwidths)
        base, remainder = divmod(self.n_features, n_bw)
        features_per_scale = [base + (1 if i < remainder else 0) for i in range(n_bw)]

        omega_list = []
        bias_list = []

        for bw, n_feat in zip(bandwidths, features_per_scale):
            omega = sample_spectral_frequencies(
                n_features=n_feat,
                n_dims=n_dims,
                bandwidth=bw,
                kernel=self.kernel,
                rng=rng,
            )
            bias = rng.uniform(0, 2 * np.pi, n_feat)

            omega_list.append(omega)
            bias_list.append(bias)

        omega_all = np.vstack(omega_list).astype(np.float64)
        bias_all = np.hstack(bias_list).astype(np.float64)
        scale = np.sqrt(2.0 / omega_all.shape[0])

        self._rff = RFFParams(omega=omega_all, bias=bias_all, scale=scale)
        self._bandwidths = bandwidths
        self._scale_offsets = np.concatenate([[0], np.cumsum(features_per_scale)])

        # Compute exact sum_z for centering correction: O(N·D), O(D) memory.
        # Must be exact: subsampled estimates have O(N²/M) bias in T
        # that destroys FPR calibration when N >> M.
        self._sum_z = compute_sum_z(coords, omega_all, bias_all, scale)

        # Estimate Z column variances via subsampling O(M·D)
        self._z_variances = compute_column_variances(
            coords, omega_all, bias_all, scale,
            max_samples=10000, random_state=self.random_state,
        )

        # Compute per-scale ||Cov_z||_F^2 and E[||z_{c,i}||^4] for kurtosis-
        # corrected Satterthwaite variance. O(M·D_s²) per scale, O(M·D²/L) total.
        self._cov_frob_sq, self._row_norm4_per_scale = compute_cov_frobenius_per_scale(
            coords, omega_all, bias_all, scale, self._scale_offsets,
            max_samples=10000, random_state=self.random_state,
        )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _normalize_expression(
        self,
        X: NDArray[np.floating] | sparse.spmatrix,
        normalize: bool | str,
        log_transform: bool = True,
        verbose: bool = False,
    ) -> NDArray[np.floating] | sparse.spmatrix:
        """Normalize expression data if needed."""
        max_val = self._max_expression_value(X)
        do_normalize = self._resolve_normalize_flag(normalize, max_val)

        X_out = X

        if do_normalize:
            if verbose:
                print(f"Normalizing expression data (max value: {max_val:.0f})...")
            X_out = normalize_total(X_out, target_sum=1e4)

        if log_transform:
            X_out = log1p_transform(X_out)

        if verbose and (do_normalize or log_transform):
            if sparse.issparse(X_out):
                max_norm = float(X_out.data.max()) if len(X_out.data) > 0 else 0
            else:
                max_norm = float(X_out.max())
            print(f"  Transformed: max value now {max_norm:.1f}")

        return X_out

    @staticmethod
    def _max_expression_value(X: NDArray[np.floating] | sparse.spmatrix) -> float:
        """Return deterministic global maximum value from dense or sparse matrix."""
        if sparse.issparse(X):
            return float(np.max(X.data)) if len(X.data) > 0 else 0.0
        X_arr = np.asarray(X)
        return float(np.max(X_arr)) if X_arr.size > 0 else 0.0

    @staticmethod
    def _resolve_normalize_flag(normalize: bool | str, max_val: float) -> bool:
        """Resolve normalize mode into a concrete boolean decision."""
        if normalize == "auto":
            return max_val > 100
        if normalize is True:
            return True
        if normalize is False:
            return False
        raise ValueError(
            f"normalize must be True, False, or 'auto', got {normalize!r}"
        )

    @staticmethod
    def _resolve_gene_names(n_genes: int, gene_names: list[str] | None) -> list[str]:
        """Return validated gene names aligned with expression columns."""
        if gene_names is None:
            return [f"Gene_{i}" for i in range(n_genes)]
        if len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names length ({len(gene_names)}) does not match "
                f"number of genes ({n_genes})"
            )
        return gene_names

    @staticmethod
    def _prepare_expression_matrix(
        X: NDArray[np.floating] | sparse.spmatrix,
    ) -> tuple[NDArray[np.floating] | sparse.csc_matrix, bool]:
        """Prepare expression matrix for fast per-gene column access."""
        if sparse.issparse(X):
            X_csc = X if sparse.isspmatrix_csc(X) else X.tocsc()
            return X_csc, True
        return np.asarray(X), False

    @staticmethod
    def _extract_gene_column(
        X: NDArray[np.floating] | sparse.csc_matrix,
        is_sparse: bool,
        gene_idx: int,
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], int]:
        """Extract non-zero row indices and values for one gene column."""
        if is_sparse:
            start, end = X.indptr[gene_idx], X.indptr[gene_idx + 1]
            row_idx = X.indices[start:end]
            values = np.asarray(X.data[start:end], dtype=np.float64)
            return row_idx, values, end - start

        col = np.asarray(X[:, gene_idx]).ravel()
        row_idx = np.nonzero(col)[0]
        values = np.asarray(col[row_idx], dtype=np.float64)
        return row_idx, values, len(row_idx)

    @staticmethod
    def _prepare_projection_kernel(
        coords: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Precompute projection-kernel matrices (or return None if singular)."""
        coords_centered = coords - coords.mean(axis=0)
        StS = coords_centered.T @ coords_centered
        try:
            return coords_centered, np.linalg.cholesky(StS)
        except np.linalg.LinAlgError:
            return coords_centered, None

    # ------------------------------------------------------------------
    # Null distribution
    # ------------------------------------------------------------------

    def _build_null_params(self, n_cells: int, has_projection: bool) -> _NullParams:
        """Build gene-independent null distribution parameters from fit() stats."""
        offsets = self._scale_offsets
        n_scales = len(self._bandwidths)
        ps_z_var = np.empty(n_scales, dtype=np.float64)
        for s in range(n_scales):
            ps_z_var[s] = np.sum(self._z_variances[offsets[s]:offsets[s + 1]])

        sum_z_var = float(np.sum(self._z_variances))
        return _NullParams(
            n_cells=n_cells,
            n_dims=self._coords.shape[1],
            mean_T_rank=n_cells * sum_z_var,
            sum_cov_frob_sq=float(np.sum(self._cov_frob_sq)),
            sum_row_norm4=float(np.sum(self._row_norm4_per_scale)),
            ps_z_var=ps_z_var,
            ps_frob_sq=self._cov_frob_sq,
            row_norm4=self._row_norm4_per_scale,
            has_projection=has_projection,
        )

    @staticmethod
    def _empty_result(
        n_genes: int,
        gene_names: list[str],
        n_expressed: NDArray[np.int64],
    ) -> FlashSResult:
        """Return all-default result when no gene passes test threshold."""
        return FlashSResult(
            gene_names=gene_names,
            pvalues=np.ones(n_genes),
            qvalues=np.ones(n_genes),
            pvalues_binary=np.ones(n_genes),
            pvalues_rank=np.ones(n_genes),
            statistics=np.zeros(n_genes),
            effect_size=np.zeros(n_genes),
            n_expressed=n_expressed,
            n_tested=0,
            tested_mask=np.zeros(n_genes, dtype=bool),
        )

    # ------------------------------------------------------------------
    # Per-gene sketch (called in the gene loop)
    # ------------------------------------------------------------------

    def _compute_gene_stats(
        self,
        row_idx: NDArray[np.intp],
        values: NDArray[np.float64],
        null: _NullParams,
        coords_centered: NDArray[np.float64],
        StS_chol: NDArray[np.float64] | None,
        return_projections: bool,
    ) -> _PerGeneStats:
        """Compute all per-gene sketch statistics for downstream inference.

        Loop-invariant state is read from ``self`` (coords, rff, scale_offsets,
        sum_z) and ``null`` (sum_cov_frob_sq, sum_row_norm4). Only per-gene
        inputs are passed explicitly.
        """
        coords = self._coords
        rff = self._rff
        offsets = self._scale_offsets
        n_cells = null.n_cells
        n_scales = len(self._bandwidths)
        n_dims = coords.shape[1]

        nnz = len(row_idx)
        ones_values = np.ones(nnz, dtype=np.float64)
        ranks = rankdata(values, method="average").astype(np.float64)

        # --- Fused triple projection: O(nnz·D), evaluate cos() once ---
        coords_nz = coords[row_idx]
        project = _project_triple_2d if n_dims == 2 else _project_triple
        v_binary_raw, v_rank_raw, v_direct_raw = project(
            ones_values, ranks, values,
            coords_nz, rff.omega, rff.bias, rff.scale,
        )

        # --- Centering correction via precomputed sum_z ---
        mean_binary = nnz / n_cells
        v_binary = v_binary_raw - mean_binary * self._sum_z

        mean_rank = float(np.sum(ranks)) / n_cells
        v_rank = v_rank_raw - mean_rank * self._sum_z

        mean_y = float(np.sum(values)) / n_cells
        v_direct = v_direct_raw - mean_y * self._sum_z

        # --- Standardization ---
        var_rank = float(np.sum(ranks**2)) / n_cells - mean_rank**2
        std_rank = 1.0
        if var_rank > 1e-10:
            std_rank = float(np.sqrt(var_rank))
            v_rank = v_rank / std_rank

        var_y = float(np.sum(values**2)) / n_cells - mean_y**2
        std_y = 1.0
        if var_y > 1e-10:
            std_y = float(np.sqrt(var_y))
            v_direct = v_direct / std_y

        # --- Excess kurtosis (for Satterthwaite variance correction) ---
        n_zero = n_cells - nnz
        y_var_binary = mean_binary * (1.0 - mean_binary)
        if y_var_binary > 1e-10:
            kappa4_binary = (1 - 6 * mean_binary + 6 * mean_binary**2) / y_var_binary
        else:
            kappa4_binary = 0.0

        kappa4_rank = (
            _excess_kurtosis_zi(ranks, mean_rank, std_rank, n_zero, n_cells)
            if var_rank > 1e-10 else 0.0
        )
        kappa4_direct = (
            _excess_kurtosis_zi(values, mean_y, std_y, n_zero, n_cells)
            if var_y > 1e-10 else 0.0
        )

        # --- Global test statistics ---
        T_binary = float(np.sum(v_binary**2))
        T_rank = float(np.sum(v_rank**2))

        # Binary Satterthwaite (global, for diagnostic p-value)
        sum_z_var = null.mean_T_rank / n_cells  # = sum(z_variances)
        mean_T_binary = y_var_binary * n_cells * sum_z_var
        var_T_binary = (
            2 * y_var_binary**2 * n_cells**2 * null.sum_cov_frob_sq
            + kappa4_binary * y_var_binary**2 * n_cells * null.sum_row_norm4
        )
        var_T_binary = max(var_T_binary, 1e-20)
        scale_binary, df_binary = _satterthwaite_params(
            mean_T_binary, var_T_binary, fallback_df=float(rff.n_features),
        )

        # --- Per-scale test statistics ---
        ps_row_b = np.empty(n_scales)
        ps_row_r = np.empty(n_scales)
        ps_row_d = np.empty(n_scales)
        for s in range(n_scales):
            sl = slice(offsets[s], offsets[s + 1])
            ps_row_b[s] = np.sum(v_binary[sl] ** 2)
            ps_row_r[s] = np.sum(v_rank[sl] ** 2)
            ps_row_d[s] = np.sum(v_direct[sl] ** 2)

        # --- Projection kernel statistics ---
        if StS_chol is not None:
            coords_nz_c = coords_centered[row_idx]

            Sty_b = coords_nz_c.sum(axis=0)
            w_proj_b = solve_triangular(StS_chol, Sty_b, lower=True)
            T_proj_binary = float(np.sum(w_proj_b**2))

            if var_rank > 1e-10:
                Sty_r = coords_nz_c.T @ (ranks / std_rank)
                w_proj_r = solve_triangular(StS_chol, Sty_r, lower=True)
                T_proj_rank = float(np.sum(w_proj_r**2))
            else:
                T_proj_rank = 0.0

            if var_y > 1e-10:
                Sty_d = coords_nz_c.T @ values / std_y
                w_proj_d = solve_triangular(StS_chol, Sty_d, lower=True)
                T_proj_direct = float(np.sum(w_proj_d**2))
            else:
                T_proj_direct = 0.0
        else:
            T_proj_binary = 0.0
            T_proj_rank = 0.0
            T_proj_direct = 0.0

        return _PerGeneStats(
            T_binary=T_binary,
            T_rank=T_rank,
            scale_binary=float(scale_binary),
            df_binary=float(df_binary),
            mean_T_binary=mean_T_binary,
            ps_T_binary=ps_row_b,
            ps_T_rank=ps_row_r,
            ps_T_direct=ps_row_d,
            y_var_binary=y_var_binary,
            kappa4_binary=kappa4_binary,
            kappa4_rank=kappa4_rank,
            kappa4_direct=kappa4_direct,
            T_proj_binary=T_proj_binary,
            T_proj_rank=T_proj_rank,
            T_proj_direct=T_proj_direct,
            rank_projection=v_rank if return_projections else None,
        )

    # ------------------------------------------------------------------
    # test()
    # ------------------------------------------------------------------

    def test(
        self,
        X: NDArray[np.floating] | sparse.spmatrix,
        gene_names: list[str] | None = None,
        verbose: bool = False,
        return_projections: bool = False,
    ) -> FlashSResult:
        """
        Test for spatial expression patterns.

        Uses sparse-aware sketching with per-gene complexity
        O(nnz·D + nnz·log nnz): O(nnz·D) for projection and
        O(nnz·log nnz) for rank transformation. No O(N) operations
        are performed per gene.

        Parameters
        ----------
        X : array-like of shape (n_cells, n_genes)
            Expression matrix (can be sparse).
        gene_names : list of str, optional
            Gene names. If None, uses indices.
        verbose : bool, default=False
            Print progress information.
        return_projections : bool, default=False
            If True, store RFF projection vectors in result.projections.
            Adds O(D) memory per gene.

        Returns
        -------
        result : FlashSResult
            Test results with P-values for all genes.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before test()")

        n_cells, n_genes = X.shape

        if n_cells != self._coords.shape[0]:
            raise ValueError(
                f"Number of cells ({n_cells}) does not match "
                f"fitted coordinates ({self._coords.shape[0]})"
            )

        gene_names = self._resolve_gene_names(n_genes, gene_names)

        # Preprocessing
        X = self._normalize_expression(X, self.normalize, self.log_transform, verbose)

        # CRITICAL: CSC gives O(1) column start/end access (vs O(N) in CSR)
        X, is_sparse = self._prepare_expression_matrix(X)

        coords_centered, StS_chol = self._prepare_projection_kernel(self._coords)
        null = self._build_null_params(n_cells, has_projection=StS_chol is not None)

        # --- Gene loop: sketch phase ---
        min_cells_test = max(self.min_expressed, _MIN_CELLS_ASYMPTOTIC)
        all_n_expressed = np.zeros(n_genes, dtype=np.int64)
        input_gene_indices: list[int] = []
        per_gene_stats: list[_PerGeneStats] = []
        projections_list: list[NDArray[np.float64]] = []

        for i in range(n_genes):
            if verbose and i % 1000 == 0:
                print(f"Testing gene {i}/{n_genes}...")

            row_idx, values, nnz = self._extract_gene_column(X, is_sparse, i)
            all_n_expressed[i] = nnz
            if nnz < min_cells_test:
                continue

            gs = self._compute_gene_stats(
                row_idx, values, null,
                coords_centered, StS_chol, return_projections,
            )

            input_gene_indices.append(i)
            per_gene_stats.append(gs)
            if return_projections:
                projections_list.append(gs.rank_projection)

        if not input_gene_indices:
            return self._empty_result(n_genes, gene_names, all_n_expressed)

        # --- Batch inference: statistics → p-values ---
        batch_gs = _GeneStats.from_list(per_gene_stats, input_gene_indices)
        pvalues, pvalues_binary, pvalues_rank, statistics, effect_sizes = (
            self._compute_kernel_pvalues(batch_gs, null)
        )

        qvalues_tested = adjust_pvalues(pvalues, method=self.adjustment)

        # --- Scatter tested results into full input-aligned arrays ---
        tested_idx = np.array(input_gene_indices)
        full_pvalues = np.ones(n_genes)
        full_qvalues = np.ones(n_genes)
        full_pvalues_binary = np.ones(n_genes)
        full_pvalues_rank = np.ones(n_genes)
        full_statistics = np.zeros(n_genes)
        full_effect_sizes = np.zeros(n_genes)
        tested_mask = np.zeros(n_genes, dtype=bool)

        full_pvalues[tested_idx] = pvalues
        full_qvalues[tested_idx] = qvalues_tested
        full_pvalues_binary[tested_idx] = pvalues_binary
        full_pvalues_rank[tested_idx] = pvalues_rank
        full_statistics[tested_idx] = statistics
        full_effect_sizes[tested_idx] = effect_sizes
        tested_mask[tested_idx] = True

        full_projections = None
        if return_projections:
            full_projections = np.zeros((n_genes, self._rff.n_features))
            full_projections[tested_idx] = np.vstack(projections_list)

        return FlashSResult(
            gene_names=gene_names,
            pvalues=full_pvalues,
            qvalues=full_qvalues,
            pvalues_binary=full_pvalues_binary,
            pvalues_rank=full_pvalues_rank,
            statistics=full_statistics,
            effect_size=full_effect_sizes,
            n_expressed=all_n_expressed,
            n_tested=len(input_gene_indices),
            tested_mask=tested_mask,
            projections=full_projections,
        )

    # ------------------------------------------------------------------
    # Batch p-value computation (pure function)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_kernel_pvalues(
        gs: _GeneStats,
        null: _NullParams,
    ) -> tuple[
        NDArray[np.float64],  # combined_pvalues
        NDArray[np.float64],  # pvalues_binary
        NDArray[np.float64],  # pvalues_rank
        NDArray[np.float64],  # statistics
        NDArray[np.float64],  # effect_sizes
    ]:
        """Batch p-value computation: statistics → p-values via Satterthwaite + Cauchy.

        Pure function: maps accumulated per-gene statistics to final p-values.
        Combines {binary, rank, direct} × L active scales + projection kernels
        via Cauchy combination.
        """
        n_tested = len(gs.T_binary)
        n_scales = gs.ps_T_binary.shape[1]
        N = null.n_cells

        # --- Global binary p-values (diagnostic) ---
        pvalues_binary = _safe_chi2_sf(gs.T_binary, gs.scale_binary, gs.df_binary)
        pvalues_binary = np.clip(pvalues_binary, np.finfo(float).tiny, 1.0)

        # --- Global rank p-values (diagnostic) ---
        var_T_rank_all = (2 * N ** 2 * null.sum_cov_frob_sq
                          + gs.kappa4_rank * N * null.sum_row_norm4)
        scale_rank, df_rank = _satterthwaite_params(null.mean_T_rank, var_T_rank_all)
        pvalues_rank = _safe_chi2_sf(
            gs.T_rank, scale_rank, df_rank, mean_threshold=null.mean_T_rank,
        )
        pvalues_rank = np.clip(pvalues_rank, np.finfo(float).tiny, 1.0)

        # --- Multi-kernel p-value matrix for Cauchy combination ---
        n_proj = 3 if null.has_projection else 0
        active_scales = [s for s in range(n_scales) if null.ps_z_var[s] > 1e-10]
        n_active = len(active_scales)
        all_pvals = np.ones((n_tested, 3 * n_active + n_proj))

        for ki, s in enumerate(active_scales):
            # Binary: kurtosis-corrected Satterthwaite (per-gene)
            mean_T_b_s = gs.y_var_binary * N * null.ps_z_var[s]
            var_T_b_s = (2 * gs.y_var_binary ** 2 * N ** 2 * null.ps_frob_sq[s]
                         + gs.kappa4_binary * gs.y_var_binary ** 2 * N * null.row_norm4[s])
            sc_b, df_b = _satterthwaite_params(mean_T_b_s, var_T_b_s)
            all_pvals[:, ki] = _safe_chi2_sf(
                gs.ps_T_binary[:, s], sc_b, df_b, mean_threshold=mean_T_b_s,
            )

            # Rank + Direct: same E[T] (unit-variance standardized)
            mean_T_r_s = N * null.ps_z_var[s]
            if mean_T_r_s > 1e-10:
                var_T_r_s = (2 * N ** 2 * null.ps_frob_sq[s]
                             + gs.kappa4_rank * N * null.row_norm4[s])
                sc_r, df_r = _satterthwaite_params(mean_T_r_s, var_T_r_s)
                all_pvals[:, n_active + ki] = _safe_chi2_sf(
                    gs.ps_T_rank[:, s], sc_r, df_r,
                )

                var_T_d_s = (2 * N ** 2 * null.ps_frob_sq[s]
                             + gs.kappa4_direct * N * null.row_norm4[s])
                sc_d, df_d = _satterthwaite_params(mean_T_r_s, var_T_d_s)
                all_pvals[:, 2 * n_active + ki] = _safe_chi2_sf(
                    gs.ps_T_direct[:, s], sc_d, df_d,
                )

        # --- Projection kernel p-values ---
        if null.has_projection:
            safe_y_var = np.where(gs.y_var_binary > 1e-10, gs.y_var_binary, 1.0)
            T_proj_b_scaled = np.where(
                gs.y_var_binary > 1e-10,
                gs.T_proj_binary / safe_y_var,
                0.0,
            )
            all_pvals[:, 3 * n_active] = _chi2_dist.sf(T_proj_b_scaled, null.n_dims)
            all_pvals[:, 3 * n_active + 1] = _chi2_dist.sf(gs.T_proj_rank, null.n_dims)
            all_pvals[:, 3 * n_active + 2] = _chi2_dist.sf(gs.T_proj_direct, null.n_dims)

        all_pvals = np.clip(all_pvals, np.finfo(float).tiny, 1.0)

        # --- Cauchy combination across all kernels ---
        combined_pvalues = batch_cauchy_combination(all_pvals)

        # --- Statistics and effect sizes ---
        statistics = gs.T_binary + gs.T_rank
        safe_mean_binary = np.where(gs.mean_T_binary > 1e-10, gs.mean_T_binary, 1.0)
        eff_binary = np.where(gs.mean_T_binary > 1e-10, gs.T_binary / safe_mean_binary, 1.0)
        safe_mean_rank = null.mean_T_rank if null.mean_T_rank > 1e-10 else 1.0
        eff_rank = np.where(null.mean_T_rank > 1e-10, gs.T_rank / safe_mean_rank, 1.0)
        effect_sizes = np.maximum(eff_binary, eff_rank)

        return combined_pvalues, pvalues_binary, pvalues_rank, statistics, effect_sizes

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit_test(
        self,
        coords: NDArray[np.floating],
        X: NDArray[np.floating] | sparse.spmatrix,
        gene_names: list[str] | None = None,
        verbose: bool = False,
        return_projections: bool = False,
    ) -> FlashSResult:
        """Fit coordinates and test expression in one call."""
        return self.fit(coords).test(X, gene_names, verbose, return_projections)
