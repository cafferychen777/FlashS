# FlashS: SVG Detection Methods

## Overview

FlashS detects spatially variable genes (SVGs) by testing the dependence between gene expression and spatial location using a nonparametric kernel-based framework. The method maps spatial coordinates to a Random Fourier Feature (RFF) space, projects each gene's expression onto this space via sparse sketching, and evaluates significance through a multi-kernel Cauchy combination across three complementary tests (binary, rank, direct) at multiple bandwidth scales plus a projection kernel. The entire procedure achieves O(nnz · D + nnz · log nnz) per-gene complexity, where nnz is the number of non-zero expression entries and D is the number of random features.

## 1. Random Fourier Features

By Bochner's theorem, a shift-invariant positive definite kernel $k(\mathbf{s}, \mathbf{s}')$ can be approximated via random features:

$$k(\mathbf{s}, \mathbf{s}') \approx \mathbf{z}(\mathbf{s})^\top \mathbf{z}(\mathbf{s}')$$

where the D-dimensional feature map is:

$$\mathbf{z}(\mathbf{s}) = \sqrt{\frac{2}{D}} \left[ \cos(\boldsymbol{\omega}_1^\top \mathbf{s} + b_1), \ldots, \cos(\boldsymbol{\omega}_D^\top \mathbf{s} + b_D) \right]^\top$$

with $\boldsymbol{\omega}_k \sim p(\boldsymbol{\omega})$ and $b_k \sim \text{Uniform}(0, 2\pi)$. For the Gaussian kernel $k(\boldsymbol{\delta}) = \exp(-\lVert\boldsymbol{\delta}\rVert^2 / 2\sigma^2)$, the spectral distribution is $p(\boldsymbol{\omega}) = \mathcal{N}(\mathbf{0}, \sigma^{-2}\mathbf{I}_d)$.

> Code: `rff.py` — `sample_spectral_frequencies()` generates the frequency vectors; feature map computed on-the-fly in `sketch.py` kernels.

## 2. Multi-Scale Bandwidth Selection

FlashS uses L Gaussian kernel bandwidths $\sigma_1 < \cdots < \sigma_L$ spanning from local to tissue-wide scales:

- **Minimum scale** $\sigma_{\min}$: median nearest-neighbor distance from a KD-tree on subsampled coordinates (M = 10,000).
- **Maximum scale** $\sigma_{\max}$: combines the spatial extent with the Fiedler value of the graph Laplacian:

$$\sigma_{\max} = \min\left(\max\left(0.25 \cdot \ell_{\text{diag}},\ 2\ell_{\text{spectral}}\right),\ 0.5 \cdot \ell_{\text{diag}}\right)$$

where $\ell_{\text{diag}}$ is the bounding box diagonal and $\ell_{\text{spectral}} = 1/\sqrt{\lambda_2}$ from the Fiedler value. A floor of $\sigma_{\max} \geq 10 \sigma_{\min}$ ensures at least one order of magnitude of scale coverage.

The L bandwidths are placed as a geometric sequence. Features are distributed across scales via `divmod(D, L)`: each scale gets ⌊D/L⌋ features, with the first D mod L scales receiving one extra feature each. This ensures exactly D total features with no loss.

> Code: `bandwidth.py` — `compute_adaptive_scales()` for bandwidth selection; `svg.py` `fit()` for divmod distribution.

## 3. Raw Count Input

FlashS operates on **raw counts** without library-size normalization or log-transformation. In spatial transcriptomics, library size is correlated with tissue structure, making it an informative component of spatial variation. Normalization removes this signal.

The binary test is naturally robust to library-size variation (presence vs. absence). The rank test is invariant to monotone transformations. Operating on raw counts preserves the full spatial signal.

## 4. SVG Detection via Sparse Sketching

### 4.1 Test Statistic

For each gene g, the test statistic is:

$$T = \lVert\tilde{\mathbf{y}}^\top \mathbf{Z}\rVert^2 = \sum_{k=1}^{D} v_k^2, \qquad v_k = \sum_{i=1}^{n} \tilde{y}_i z_k(\mathbf{s}_i)$$

where $\tilde{\mathbf{y}} = \mathbf{y} - \bar{y}\mathbf{1}$ is the centered expression vector.

### 4.2 Sparse Sketching

The projection decomposes into a sparse sum over non-zero entries and a precomputed centering term:

$$v_k = \sum_{i:\ y_i \neq 0} y_i\, z_k(\mathbf{s}_i) - \bar{y}\, \zeta_k$$

where the sparse sum costs O(nnz_g) and $\zeta_k = \sum_{i=1}^{n} z_k(\mathbf{s}_i)$ is precomputed.

The column sum vector $\boldsymbol{\zeta} \in \mathbb{R}^D$ is shared across all genes, precomputed exactly once during `fit()`. This is an O(N · D) pass with O(D) memory (no matrix materialization); each $z_k(\mathbf{s}_i) = \sqrt{2/D}\cos(\boldsymbol{\omega}_k^\top \mathbf{s}_i + b_k)$ is computed on-the-fly from stored parameters. Exact computation is necessary: a subsampled estimate introduces O(N/√M) error per feature, which enters the test statistic quadratically as O(N²/M), overwhelming the null expectation O(N) when N ≫ M.

All three projections (binary, rank, direct) are computed in a single fused pass via `_project_triple_2d` (2D) or `_project_triple` (general dimensions), both Numba JIT-compiled and parallelized over D. The cos() evaluation is shared across all three channels per (feature, cell) pair.

> Code: `sketch.py` — `compute_sum_z()` computes the centering vector exactly; `_project_triple_2d` / `_project_triple` for fused projection.

### 4.3 Three-Part Test

**Binary test** — tests whether the spatial pattern of expression presence deviates from randomness:

$$\mathbf{v}^{(b)} = \sum_{i \in \mathcal{I}_g} \mathbf{z}(\mathbf{s}_i) - \frac{|\mathcal{I}_g|}{n}\,\boldsymbol{\zeta}, \qquad T^{(b)} = \lVert\mathbf{v}^{(b)}\rVert^2$$

where $\mathcal{I}_g = \lbrace i : y_{ig} > 0 \rbrace$.

**Rank test** — tests whether expression intensity among expressing cells is spatially structured:

$$\mathbf{v}^{(r)} = \frac{1}{\hat{\sigma}_r}\left(\sum_{i \in \mathcal{I}_g} r_i\,\mathbf{z}(\mathbf{s}_i) - \bar{r}\,\boldsymbol{\zeta}\right), \qquad T^{(r)} = \lVert\mathbf{v}^{(r)}\rVert^2$$

where $r_i = \text{rank}(y_{ig} \mid i \in \mathcal{I}_g)$ (ranks within non-zeros only), $\bar{r} = n^{-1}\sum_{i \in \mathcal{I}_g} r_i$ (zero-padded mean), and $\hat{\sigma}_r^2 = n^{-1}\sum_{i \in \mathcal{I}_g} r_i^2 - \bar{r}^2$.

**Direct test** — tests whether raw expression values exhibit spatial structure:

$$\mathbf{v}^{(d)} = \frac{1}{\hat{\sigma}_y}\left(\sum_{i \in \mathcal{I}_g} y_i\,\mathbf{z}(\mathbf{s}_i) - \bar{y}\,\boldsymbol{\zeta}\right), \qquad T^{(d)} = \lVert\mathbf{v}^{(d)}\rVert^2$$

where $\bar{y} = n^{-1}\sum_{i \in \mathcal{I}_g} y_i$ (zero-padded mean), and $\hat{\sigma}_y^2 = n^{-1}\sum_{i \in \mathcal{I}_g} y_i^2 - \bar{y}^2$.

**Small sample handling**: Genes with fewer than max(`min_expressed`, 30) expressing cells are excluded from testing (p = 1, `tested_mask` = False). The 30-cell floor ensures chi-square asymptotic validity via CLT convergence.

### 4.4 Null Distribution

Under H₀, $T \sim \kappa \chi^2_\nu$ via Satterthwaite's moment matching:

$$\kappa = \frac{\text{Var}(T)}{2\,\mathbb{E}[T]}, \qquad \nu = \frac{2\,\mathbb{E}[T]^2}{\text{Var}(T)}$$

**Binary test**: $\text{Var}(y^{(b)}) = \bar{p}(1-\bar{p})$ where $\bar{p} = |\mathcal{I}_g|/n$:

$$\mathbb{E}[T^{(b)}] = \bar{p}(1-\bar{p}) \cdot n \sum_{k=1}^D \sigma^2_{z_k}$$

$$\text{Var}(T^{(b)}) = 2\bar{p}^2(1-\bar{p})^2 \cdot n^2 \lVert\text{Cov}_z\rVert_F^2$$

where $\sigma^2_{z_k} = \text{Var}(z_k(\mathbf{s}))$ estimated via M = 10,000 subsampled coordinates and $\lVert\text{Cov}_z\rVert_F^2 = \sum_{j,k} \text{Cov}(z_j, z_k)^2$ is the squared Frobenius norm of the Z-column covariance matrix (also estimated via subsampling). When Z columns are independent this reduces to $\sum_k \sigma^4_{z_k}$, but RFF features at large bandwidths are correlated, making the full Frobenius norm necessary for accurate null variance estimation. Parameters are **gene-specific** (different $\bar{p}$).

**Rank and direct tests**: After standardization, Var(y) = 1, so null distribution parameters are **shared** across genes within each test type.

## 5. Multi-Kernel Cauchy Combination

### 5.1 Per-Scale P-values

The D features are partitioned into L groups per bandwidth scale. For each scale ℓ and each test type $t \in \lbrace b, r, d \rbrace$:

$$T_\ell^{(t)} = \sum_{k \in S_\ell} \left(v_k^{(t)}\right)^2$$

with per-scale null distribution parameters:

$$\kappa_\ell = \frac{\text{Var}(y^{(t)}) \cdot n \,\lVert\text{Cov}_{z,\ell}\rVert_F^2}{\sum_{k \in S_\ell} \sigma^2_{z_k}}, \qquad \nu_\ell = \frac{\left(\sum_{k \in S_\ell} \sigma^2_{z_k}\right)^2}{\lVert\text{Cov}_{z,\ell}\rVert_F^2}$$

where $\lVert\text{Cov}_{z,\ell}\rVert_F^2$ is the squared Frobenius norm of the covariance among features in scale ℓ. Only **active scales** (those with $\sum_{k \in S_\ell}\sigma^2_{z_k} > \epsilon$) contribute. Scales with zero variance are excluded to avoid contributing uninformative p = 1 kernels.

### 5.2 Projection Kernel

Tests for linear spatial gradients using centered coordinates $\tilde{\mathbf{S}}$:

$$T_{\text{proj}}^{(t)} = (\tilde{\mathbf{S}}^\top \mathbf{y}^{(t)})^\top (\tilde{\mathbf{S}}^\top \tilde{\mathbf{S}})^{-1} (\tilde{\mathbf{S}}^\top \mathbf{y}^{(t)})$$

Under H₀: $T_{\text{proj}}^{(t)} / \text{Var}(y^{(t)}) \sim \chi^2_d$. Computed via Cholesky: $T_{\text{proj}} = \lVert\mathbf{L}^{-1}\tilde{\mathbf{S}}^\top \mathbf{y}\rVert^2$ where $\tilde{\mathbf{S}}^\top\tilde{\mathbf{S}} = \mathbf{L}\mathbf{L}^\top$.

If Cholesky fails (degenerate coordinates), projection kernels are omitted.

### 5.3 Cauchy Combination Across All Kernels

The K p-values from all kernel × test-type combinations are combined with equal weights:

$$K = 3 \times L_{\text{active}} + n_{\text{proj}}$$

where $L_{\text{active}}$ is the number of active scales and $n_{\text{proj}} \in \lbrace 0, 3 \rbrace$ (3 projection kernels for binary/rank/direct, or 0 if degenerate).

$$T_{\text{CCT}} = \frac{1}{K}\sum_{j=1}^{K} \tan\left(\left(\frac{1}{2} - p_j\right)\pi\right), \qquad p_{\text{final}} = \frac{1}{2} - \frac{\arctan(T_{\text{CCT}})}{\pi}$$

> Code: `pvalue.py` — `batch_cauchy_combination()` for vectorized Cauchy combination.

## 6. Effect Size and Scoring

Effect size:

$$e_g = \max\left(\frac{T_g^{(b)}}{\mathbb{E}[T_g^{(b)} \mid H_0]},\ \frac{T_g^{(r)}}{\mathbb{E}[T_g^{(r)} \mid H_0]}\right)$$

## 7. Computational Complexity

| Operation | Time | Memory |
|-----------|------|--------|
| Model fitting (centering + variance + covariance) | O(N·D + M·D + M·D²/L) | O(M·D/L) |
| Per-gene projection | O(nnz_g · D) | O(D) |
| Per-gene rank transform | O(nnz_g · log nnz_g) | O(nnz_g) |
| Per-gene p-value | O(K) | O(K) |
| All genes total | O(nnz · D) | O(G) |

N = cells, D = 500 (default), L = 7 (default), M = 10,000 (subsampling), G = genes.

## 8. Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_features` | 500 | Total RFF features D |
| `n_scales` | 7 | Bandwidth scales L |
| `min_expressed` | 5 | Minimum expressing cells; effective threshold is max(`min_expressed`, 30) |
| `normalize` | False | No library-size normalization |
| `log_transform` | False | No log1p transformation |
| `adjustment` | "storey" | Storey's q-value correction |
| `random_state` | 0 | Reproducibility seed |

## References

- Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines. NeurIPS.
- Liu & Xie (2020). Cauchy Combination Test. JASA.
- Zhu, Shang & Li (2021). SPARK-X. Genome Biology.
- Satterthwaite (1946). Approximate Distribution of Estimates of Variance Components. Biometrics Bulletin.
