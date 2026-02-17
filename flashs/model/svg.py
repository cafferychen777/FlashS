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
    compute_batch_pvalues,
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
    adjustment : str, default="storey"
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
        adjustment: Literal["bh", "bonferroni", "holm", "by", "storey", "none"] = "storey",
        random_state: int | None = 0,
    ):
        # Validate parameters
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

        # Store parameters (bandwidth already normalized to list[float] | None)
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
        self._coords: NDArray[np.float64]

    @property
    def bandwidths(self) -> list[float]:
        """Bandwidth values used for multi-scale RFF."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._bandwidths

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
            coords,
            omega_all,
            bias_all,
            scale,
            max_samples=10000,
            random_state=self.random_state,
        )

        # Compute per-scale ||Cov_z||_F^2 for corrected Satterthwaite variance.
        # RFF features at large bandwidths are correlated, so the standard
        # sum(Var(z_k)^2) underestimates the null variance. This correction
        # uses the full covariance structure: O(M·D_s²) per scale, O(M·D²/L) total.
        self._cov_frob_sq = compute_cov_frobenius_per_scale(
            coords,
            omega_all,
            bias_all,
            scale,
            self._scale_offsets,
            max_samples=10000,
            random_state=self.random_state,
        )

        self._fitted = True
        return self

    def _normalize_expression(
        self,
        X: NDArray[np.floating] | sparse.spmatrix,
        normalize: bool | str,
        log_transform: bool = True,
        verbose: bool = False,
    ) -> NDArray[np.floating] | sparse.spmatrix:
        """
        Normalize expression data if needed.

        Parameters
        ----------
        X : array-like
            Expression matrix (can be sparse).
        normalize : bool or "auto"
            - "auto": Normalize if max > 100 (likely raw counts)
            - True: Always normalize
            - False: Never normalize
        log_transform : bool, default=True
            Whether to apply log1p transformation.
        verbose : bool
            Print normalization info.

        Returns
        -------
        X_normalized : array-like
            Normalized expression matrix.
        """
        # Check if normalization is needed
        is_sparse = sparse.issparse(X)
        if is_sparse:
            sample_vals = X.data[:min(10000, len(X.data))] if len(X.data) > 0 else np.array([0])
        else:
            sample_vals = np.asarray(X).ravel()[:min(10000, X.size)]

        max_val = float(np.max(sample_vals)) if len(sample_vals) > 0 else 0
        is_raw_counts = max_val > 100

        # Determine whether to normalize
        if normalize == "auto":
            do_normalize = is_raw_counts
        elif normalize is True:
            do_normalize = True
        else:
            do_normalize = False

        # Apply transformations
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
        Achieves this by:
        1. Converting to CSC format for O(1) column access
        2. Using sparse indices directly (no toarray())
        3. Centering via precomputed sum_z correction

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
            These vectors capture each gene's "spatial shape fingerprint"
            and can be used for clustering genes by spatial pattern.
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

        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(n_genes)]
        elif len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names length ({len(gene_names)}) does not match "
                f"number of genes ({n_genes})"
            )

        # Preprocessing
        X = self._normalize_expression(X, self.normalize, self.log_transform, verbose)

        # CRITICAL: Convert to CSC for O(1) column access
        # CSR column access is O(N), CSC is O(nnz_col)
        if sparse.issparse(X):
            if not sparse.isspmatrix_csc(X):
                X = X.tocsc()
            # Get direct access to CSC internals
            indptr = X.indptr
            indices = X.indices
            data = X.data
            is_sparse = True
        else:
            X = np.asarray(X)
            is_sparse = False

        rff = self._rff
        coords = self._coords
        n_features = rff.n_features

        # Pre-compute constants used in null distribution calculation
        sum_z_var = np.sum(self._z_variances)
        # Use corrected Frobenius norm for variance (accounts for Z column correlations)
        sum_cov_frob_sq = np.sum(self._cov_frob_sq)

        # Pre-compute rank test null distribution parameters (y_var_rank = 1.0)
        mean_T_rank_const = n_cells * sum_z_var
        var_T_rank_const = 2 * n_cells ** 2 * sum_cov_frob_sq
        if mean_T_rank_const > 1e-10 and var_T_rank_const > 1e-10:
            scale_rank_const = var_T_rank_const / (2 * mean_T_rank_const)
            df_rank_const = 2 * mean_T_rank_const ** 2 / var_T_rank_const
        else:
            scale_rank_const = 1.0
            df_rank_const = float(n_features)

        # === Per-scale null distribution parameters ===
        n_scales = len(self._bandwidths)
        offsets = self._scale_offsets
        ps_z_var = np.zeros(n_scales)        # per-scale sum of z_variances (for E[T])
        ps_frob_sq = self._cov_frob_sq       # per-scale ||Cov_z||_F^2 (for Var[T])
        for s in range(n_scales):
            z_s = self._z_variances[offsets[s]:offsets[s + 1]]
            ps_z_var[s] = np.sum(z_s)

        # Per-scale rank null constants (shared across genes)
        # Corrected: use ||Cov_z||_F^2 instead of sum(Var(z_k)^2) for variance
        safe_z_var = np.where(ps_z_var > 1e-10, ps_z_var, 1.0)
        safe_frob_sq = np.where(ps_frob_sq > 1e-10, ps_frob_sq, 1.0)
        ps_scale_rank = np.where(
            ps_z_var > 1e-10,
            n_cells * ps_frob_sq / safe_z_var,
            1.0,
        )
        ps_df_rank = np.where(
            ps_frob_sq > 1e-10,
            ps_z_var ** 2 / safe_frob_sq,
            1.0,
        )

        # === Projection kernel pre-computation ===
        # Linear test: tests spatial gradients directly (like SPARK-X's projection kernel)
        coords_centered = coords - coords.mean(axis=0)
        StS = coords_centered.T @ coords_centered  # (d, d)
        try:
            StS_chol = np.linalg.cholesky(StS)
            has_projection = True
        except np.linalg.LinAlgError:
            has_projection = False

        # Collect statistics for batch p-value computation
        # Lists for genes that pass filtering
        tested_gene_names: list[str] = []
        input_gene_indices: list[int] = []  # maps tested position → input position
        all_n_expressed = np.zeros(n_genes, dtype=np.int64)  # for ALL genes
        T_binary_list: list[float] = []
        T_rank_list: list[float] = []
        scale_binary_list: list[float] = []
        df_binary_list: list[float] = []
        mean_T_binary_list: list[float] = []
        projections_list: list[NDArray[np.float64]] = [] if return_projections else None

        # Per-scale statistics (dynamic, converted to arrays after loop)
        ps_T_binary_list: list[NDArray[np.float64]] = []
        ps_T_rank_list: list[NDArray[np.float64]] = []
        ps_T_direct_list: list[NDArray[np.float64]] = []
        ps_y_var_binary_list: list[float] = []

        # Projection kernel statistics
        T_proj_binary_list: list[float] = []
        T_proj_rank_list: list[float] = []
        T_proj_direct_list: list[float] = []

        # Minimum cells for asymptotic chi-square validity (CLT threshold)
        _MIN_CELLS_ASYMPTOTIC = 30
        min_cells_test = max(self.min_expressed, _MIN_CELLS_ASYMPTOTIC)

        for i in range(n_genes):
            if verbose and i % 1000 == 0:
                print(f"Testing gene {i}/{n_genes}...")

            # Extract sparse column - O(1) for CSC!
            if is_sparse:
                start, end = indptr[i], indptr[i + 1]
                row_idx = indices[start:end]
                values = np.asarray(data[start:end], dtype=np.float64)
                nnz = end - start
            else:
                # Dense fallback
                col = X[:, i]
                row_idx = np.nonzero(col)[0]
                values = np.asarray(col[row_idx], dtype=np.float64)
                nnz = len(row_idx)

            # Skip genes without enough expressing cells for asymptotic validity
            all_n_expressed[i] = nnz
            if nnz < min_cells_test:
                continue

            # === Fused triple projection (binary + rank + direct in one pass) ===
            ones_values = np.ones(nnz, dtype=np.float64)
            ranks = rankdata(values, method="average").astype(np.float64)

            coords_nz = coords[row_idx]
            if coords_nz.shape[1] == 2:
                v_binary_raw, v_rank_raw, v_direct_raw = _project_triple_2d(
                    ones_values, ranks, values, coords_nz,
                    rff.omega, rff.bias, rff.scale,
                )
            else:
                # Generic n-D: single fused pass
                v_binary_raw, v_rank_raw, v_direct_raw = _project_triple(
                    ones_values, ranks, values, coords_nz,
                    rff.omega, rff.bias, rff.scale,
                )

            # Apply centering correction
            mean_binary = nnz / n_cells
            v_binary = v_binary_raw - mean_binary * self._sum_z

            sum_ranks = np.sum(ranks)
            mean_rank = sum_ranks / n_cells
            v_rank = v_rank_raw - mean_rank * self._sum_z

            mean_y = np.sum(values) / n_cells
            v_direct = v_direct_raw - mean_y * self._sum_z

            # Standardize rank
            var_rank = np.sum(ranks**2) / n_cells - mean_rank**2
            if var_rank > 1e-10:
                std_rank = np.sqrt(var_rank)
                v_rank = v_rank / std_rank

            # Standardize direct
            var_y = np.sum(values ** 2) / n_cells - mean_y ** 2
            if var_y > 1e-10:
                v_direct = v_direct / np.sqrt(var_y)

            # Test statistics
            T_binary = float(np.sum(v_binary ** 2))
            T_rank = float(np.sum(v_rank ** 2))

            # Null distribution parameters
            y_var_binary = mean_binary * (1 - mean_binary)
            mean_T_binary = y_var_binary * n_cells * sum_z_var
            var_T_binary = 2 * y_var_binary ** 2 * n_cells ** 2 * sum_cov_frob_sq

            if mean_T_binary > 1e-10 and var_T_binary > 1e-10:
                scale_binary = var_T_binary / (2 * mean_T_binary)
                df_binary = 2 * mean_T_binary ** 2 / var_T_binary
            else:
                scale_binary = 1.0
                df_binary = float(n_features)

            # Collect statistics
            tested_gene_names.append(gene_names[i])
            input_gene_indices.append(i)
            T_binary_list.append(T_binary)
            T_rank_list.append(T_rank)
            scale_binary_list.append(scale_binary)
            df_binary_list.append(df_binary)
            mean_T_binary_list.append(mean_T_binary)
            if return_projections:
                projections_list.append(v_rank)

            # === Per-scale test statistics ===
            ps_row_b = np.empty(n_scales)
            ps_row_r = np.empty(n_scales)
            ps_row_d = np.empty(n_scales)
            for s in range(n_scales):
                sl = slice(offsets[s], offsets[s + 1])
                ps_row_b[s] = np.sum(v_binary[sl] ** 2)
                ps_row_r[s] = np.sum(v_rank[sl] ** 2)
                ps_row_d[s] = np.sum(v_direct[sl] ** 2)
            ps_T_binary_list.append(ps_row_b)
            ps_T_rank_list.append(ps_row_r)
            ps_T_direct_list.append(ps_row_d)
            ps_y_var_binary_list.append(y_var_binary)

            # === Projection kernel test statistic ===
            # Tests linear spatial gradients: T = (S^T y)^T (S^T S)^{-1} (S^T y)
            # Since coords are centered, sum(S)=0, so centering correction vanishes.
            if has_projection:
                coords_nz_c = coords_centered[row_idx]

                Sty_b = coords_nz_c.sum(axis=0)
                w_proj_b = solve_triangular(StS_chol, Sty_b, lower=True)
                T_proj_binary_list.append(float(np.sum(w_proj_b ** 2)))

                if var_rank > 1e-10:
                    Sty_r = coords_nz_c.T @ (ranks / std_rank)
                    w_proj_r = solve_triangular(StS_chol, Sty_r, lower=True)
                    T_proj_rank_list.append(float(np.sum(w_proj_r ** 2)))
                else:
                    T_proj_rank_list.append(0.0)

                if var_y > 1e-10:
                    Sty_d = coords_nz_c.T @ values / np.sqrt(var_y)
                    w_proj_d = solve_triangular(StS_chol, Sty_d, lower=True)
                    T_proj_direct_list.append(float(np.sum(w_proj_d ** 2)))
                else:
                    T_proj_direct_list.append(0.0)
            else:
                T_proj_binary_list.append(0.0)
                T_proj_rank_list.append(0.0)
                T_proj_direct_list.append(0.0)

        if len(tested_gene_names) == 0:
            return FlashSResult(
                gene_names=gene_names,
                pvalues=np.ones(n_genes),
                qvalues=np.ones(n_genes),
                pvalues_binary=np.ones(n_genes),
                pvalues_rank=np.ones(n_genes),
                statistics=np.zeros(n_genes),
                effect_size=np.zeros(n_genes),
                n_expressed=all_n_expressed,
                n_tested=0,
                tested_mask=np.zeros(n_genes, dtype=bool),
            )

        # === Batch p-value computation (~200x faster than per-gene) ===
        T_binary_arr = np.array(T_binary_list)
        T_rank_arr = np.array(T_rank_list)
        scale_binary_arr = np.array(scale_binary_list)
        df_binary_arr = np.array(df_binary_list)
        mean_T_binary_arr = np.array(mean_T_binary_list)

        # Binary p-values: each gene has different scale/df
        scaled_T_binary = T_binary_arr / scale_binary_arr
        pvalues_binary = _chi2_dist.sf(scaled_T_binary, df_binary_arr)
        pvalues_binary = np.clip(pvalues_binary, np.finfo(float).tiny, 1.0)

        # Rank p-values: all use same scale/df (precomputed constants)
        pvalues_rank = compute_batch_pvalues(T_rank_arr, scale_rank_const, df_rank_const)

        # === Multiscale Cauchy combination (multi-kernel style) ===
        # Full multi-kernel: {binary, rank, direct} × n_scales per-scale p-values
        # plus {binary, rank, direct} projection kernel p-values.
        # Cauchy combination adapts to the best-performing kernels automatically.
        # Under H0, non-informative tests contribute ~0 to Cauchy statistic.
        n_tested_genes = len(tested_gene_names)
        ps_T_binary_arr = np.array(ps_T_binary_list)
        ps_T_rank_arr = np.array(ps_T_rank_list)
        ps_T_direct_arr = np.array(ps_T_direct_list)
        ps_y_var_binary_arr = np.array(ps_y_var_binary_list)
        T_proj_binary_arr = np.array(T_proj_binary_list)
        T_proj_rank_arr = np.array(T_proj_rank_list)
        T_proj_direct_arr = np.array(T_proj_direct_list)

        n_dims = coords.shape[1]
        n_proj = 3 if has_projection else 0

        # Only include scales that have actual features (skip 0-feature scales)
        active_scales = [s for s in range(n_scales) if ps_z_var[s] > 1e-10]
        n_active = len(active_scales)
        n_kernels = 3 * n_active + n_proj
        all_pvals = np.ones((n_tested_genes, n_kernels))

        for ki, s in enumerate(active_scales):
            # Binary: scale depends on per-gene y_var_binary
            ps_scale_b = ps_y_var_binary_arr * n_cells * ps_frob_sq[s] / ps_z_var[s]
            ps_df_b = ps_z_var[s] ** 2 / ps_frob_sq[s]
            safe_scale_b = np.where(ps_scale_b > 1e-10, ps_scale_b, 1.0)
            scaled_T_b = np.where(
                ps_scale_b > 1e-10,
                ps_T_binary_arr[:, s] / safe_scale_b,
                0.0,
            )
            all_pvals[:, ki] = _chi2_dist.sf(scaled_T_b, ps_df_b)

            # Rank: gene-independent scale (precomputed)
            scaled_T_r = ps_T_rank_arr[:, s] / ps_scale_rank[s]
            all_pvals[:, n_active + ki] = _chi2_dist.sf(
                scaled_T_r, ps_df_rank[s]
            )

            # Direct: same null distribution as rank (standardized var=1)
            scaled_T_d = ps_T_direct_arr[:, s] / ps_scale_rank[s]
            all_pvals[:, 2 * n_active + ki] = _chi2_dist.sf(
                scaled_T_d, ps_df_rank[s]
            )

        if has_projection:
            # Projection binary: T / y_var ~ chi2(d) under H0
            safe_y_var = np.where(
                ps_y_var_binary_arr > 1e-10, ps_y_var_binary_arr, 1.0
            )
            T_proj_b_scaled = np.where(
                ps_y_var_binary_arr > 1e-10,
                T_proj_binary_arr / safe_y_var,
                0.0,
            )
            all_pvals[:, 3 * n_active] = _chi2_dist.sf(T_proj_b_scaled, n_dims)

            # Projection rank: T already standardized ~ chi2(d)
            all_pvals[:, 3 * n_active + 1] = _chi2_dist.sf(T_proj_rank_arr, n_dims)

            # Projection direct: T already standardized ~ chi2(d)
            all_pvals[:, 3 * n_active + 2] = _chi2_dist.sf(T_proj_direct_arr, n_dims)

        all_pvals = np.clip(all_pvals, np.finfo(float).tiny, 1.0)

        # Cauchy combination across all kernels
        pvalues = batch_cauchy_combination(all_pvals)

        # Statistics and effect sizes
        statistics = T_binary_arr + T_rank_arr
        safe_mean_T_binary = np.where(mean_T_binary_arr > 1e-10, mean_T_binary_arr, 1.0)
        eff_binary = np.where(
            mean_T_binary_arr > 1e-10,
            T_binary_arr / safe_mean_T_binary,
            1.0
        )
        safe_mean_T_rank = mean_T_rank_const if mean_T_rank_const > 1e-10 else 1.0
        eff_rank = np.where(
            mean_T_rank_const > 1e-10,
            T_rank_arr / safe_mean_T_rank,
            1.0
        )
        effect_sizes = np.maximum(eff_binary, eff_rank)

        # FDR correction on tested genes only (bioinformatics convention:
        # correct for the number of tests actually performed)
        qvalues_tested = adjust_pvalues(pvalues, method=self.adjustment)

        # Scatter tested results into full input-aligned arrays
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

        # Build projection matrix if requested
        full_projections = None
        if return_projections:
            full_projections = np.zeros((n_genes, n_features))
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
            n_tested=len(tested_gene_names),
            tested_mask=tested_mask,
            projections=full_projections,
        )

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
