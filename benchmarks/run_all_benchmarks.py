"""
Benchmark suite for FlashS after Satterthwaite variance fix.

Runs four experiments:
1. Type I error control (null simulations)
2. Power analysis (SVG detection across patterns/effects/sparsity)
3. Ablation over D (n_features)
4. Ablation over n_scales

All results saved to benchmarks/results/*.csv
"""

import sys
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse

# Use local FlashS package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flashs import FlashS


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def make_grid(grid_size: int) -> np.ndarray:
    """Create a 2D grid of coordinates."""
    x = np.arange(grid_size)
    xx, yy = np.meshgrid(x, x)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return coords


def simulate_negbin(n_spots: int, n_genes: int, r: float = 1.0, p: float = 0.1,
                    rng: np.random.Generator = None) -> sparse.csc_matrix:
    """
    Simulate expression from Negative Binomial(r, p).

    NB parameterized so that mean = r*p/(1-p) and var = r*p/(1-p)^2.
    With r=1, p=0.1: mean ~0.111, highly sparse.
    """
    if rng is None:
        rng = np.random.default_rng()
    # scipy NB: number of successes r, probability of success p_scipy = 1-p
    counts = rng.negative_binomial(r, 1 - p, size=(n_spots, n_genes))
    return sparse.csc_matrix(counts.astype(np.float64))


def add_spatial_pattern(
    base_expr: np.ndarray,
    coords: np.ndarray,
    pattern: str,
    effect_size: float,
    grid_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add spatial pattern to base expression for SVG genes.

    Parameters
    ----------
    base_expr : ndarray (n_spots,)
        Base expression values (dense).
    coords : ndarray (n_spots, 2)
        Spatial coordinates.
    pattern : str
        One of 'hotspot', 'gradient', 'periodic'.
    effect_size : float
        Strength of the spatial signal.
    grid_size : int
        Size of the grid (for normalization).
    rng : Generator
        Random number generator (unused, for API consistency).

    Returns
    -------
    modified : ndarray (n_spots,)
        Expression with added spatial component.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    if pattern == "hotspot":
        # Gaussian envelope centered at (grid_size/2, grid_size/2) with radius grid_size/5
        cx, cy = grid_size / 2.0, grid_size / 2.0
        radius = grid_size / 5.0
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        spatial = np.exp(-dist_sq / (2 * radius ** 2))
    elif pattern == "gradient":
        # Linear gradient along x-axis
        spatial = x / grid_size
    elif pattern == "periodic":
        # Sinusoidal pattern: sin(2*pi*3*x/grid_size)
        spatial = np.sin(2 * np.pi * 3 * x / grid_size) * 0.5 + 0.5
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Scale spatial component by effect_size and add
    # Use multiplicative effect: expr * (1 + effect * spatial)
    modified = base_expr * (1.0 + effect_size * spatial)
    return np.maximum(modified, 0.0)


def simulate_svg_data(
    grid_size: int,
    n_svg: int,
    n_null: int,
    pattern: str,
    effect_size: float,
    sparsity: float,
    rng: np.random.Generator,
) -> tuple[sparse.csc_matrix, list[str], list[bool]]:
    """
    Simulate expression matrix with known SVGs and null genes.

    Parameters
    ----------
    grid_size : int
        Grid dimension (total spots = grid_size^2).
    n_svg : int
        Number of spatially variable genes.
    n_null : int
        Number of null (non-spatial) genes.
    pattern : str
        Spatial pattern type.
    effect_size : float
        Signal strength.
    sparsity : float
        Fraction of spots with zero expression (additional dropout).
    rng : Generator
        Random number generator.

    Returns
    -------
    X : sparse matrix (n_spots, n_genes)
        Expression matrix.
    gene_names : list[str]
        Gene names.
    is_svg : list[bool]
        True for SVG genes.
    """
    coords = make_grid(grid_size)
    n_spots = coords.shape[0]
    n_genes = n_svg + n_null

    # Base expression from NegBin(1, 0.1)
    base = rng.negative_binomial(1, 0.9, size=(n_spots, n_genes)).astype(np.float64)

    # Add spatial patterns to SVG genes
    for j in range(n_svg):
        base[:, j] = add_spatial_pattern(
            base[:, j], coords, pattern, effect_size, grid_size, rng
        )

    # Apply additional sparsity (dropout)
    if sparsity > 0:
        dropout_mask = rng.random(size=(n_spots, n_genes)) > sparsity
        base = base * dropout_mask

    X = sparse.csc_matrix(base)
    gene_names = [f"SVG_{j}" for j in range(n_svg)] + [f"Null_{j}" for j in range(n_null)]
    is_svg = [True] * n_svg + [False] * n_null

    return X, gene_names, is_svg


# ---------------------------------------------------------------------------
# Experiment 1: Type I Error Control
# ---------------------------------------------------------------------------

def run_type1_error():
    """
    Test Type I error under the null (no spatial signal).

    Grid sizes: 30x30 (900), 50x50 (2500), 100x100 (10000)
    20 reps per grid size, 1000 null genes each.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Type I Error Control")
    print("=" * 70)

    grid_sizes = [30, 50, 100]
    n_reps = 20
    n_genes = 1000
    rows = []

    for gs in grid_sizes:
        n_spots = gs * gs
        coords = make_grid(gs)
        print(f"\n  Grid {gs}x{gs} ({n_spots} spots), {n_genes} null genes, {n_reps} reps")

        for rep in range(n_reps):
            rng = np.random.default_rng(42 + rep + gs * 1000)

            # Pure null: NegBin expression, no spatial structure
            X = simulate_negbin(n_spots, n_genes, r=1.0, p=0.1, rng=rng)
            gene_names = [f"Null_{j}" for j in range(n_genes)]

            # Run FlashS with default parameters
            model = FlashS(
                n_features=500,
                n_scales=7,
                normalize=False,
                log_transform=False,
                random_state=rep,
            )
            result = model.fit(coords).test(X, gene_names=gene_names)

            n_tested = result.n_tested
            n_sig_p05 = int(np.sum(result.pvalues < 0.05))
            n_sig_q05 = int(np.sum(result.qvalues < 0.05))
            fpr_p = n_sig_p05 / max(n_tested, 1)
            fdr_q = n_sig_q05 / max(n_tested, 1)

            rows.append({
                "grid_size": gs,
                "n_spots": n_spots,
                "rep": rep,
                "n_genes": n_genes,
                "n_tested": n_tested,
                "n_significant_p05": n_sig_p05,
                "n_significant_q05": n_sig_q05,
                "fpr_pvalue": fpr_p,
                "fdr_qvalue": fdr_q,
            })

            if (rep + 1) % 5 == 0:
                print(f"    Rep {rep+1}/{n_reps}: tested={n_tested}, "
                      f"FPR(p<0.05)={fpr_p:.4f}, FDR(q<0.05)={fdr_q:.4f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "results", "type1_error.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Summary
    print("\n  Summary by grid size:")
    for gs in grid_sizes:
        sub = df[df["grid_size"] == gs]
        print(f"    {gs}x{gs}: mean FPR(p<0.05)={sub['fpr_pvalue'].mean():.4f} "
              f"(+/- {sub['fpr_pvalue'].std():.4f}), "
              f"mean FDR(q<0.05)={sub['fdr_qvalue'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Experiment 2: Power Analysis
# ---------------------------------------------------------------------------

def run_power_analysis():
    """
    Power analysis across patterns, effect sizes, and sparsity levels.

    50x50 grid, 500 SVG + 500 null genes.
    Patterns: hotspot, gradient, periodic
    Effect sizes: 0.1, 0.3, 0.5
    Sparsity: 0.1, 0.3, 0.5
    10 reps each.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Power Analysis")
    print("=" * 70)

    grid_size = 50
    n_svg = 500
    n_null = 500
    patterns = ["hotspot", "gradient", "periodic"]
    effect_sizes = [0.1, 0.3, 0.5]
    sparsities = [0.1, 0.3, 0.5]
    n_reps = 10
    rows = []

    coords = make_grid(grid_size)
    total_configs = len(patterns) * len(effect_sizes) * len(sparsities)
    config_idx = 0

    for pattern in patterns:
        for effect in effect_sizes:
            for sparsity in sparsities:
                config_idx += 1
                print(f"\n  Config {config_idx}/{total_configs}: "
                      f"pattern={pattern}, effect={effect}, sparsity={sparsity}")

                for rep in range(n_reps):
                    rng = np.random.default_rng(42 + rep + hash((pattern, effect, sparsity)) % 10000)

                    X, gene_names, is_svg = simulate_svg_data(
                        grid_size=grid_size,
                        n_svg=n_svg,
                        n_null=n_null,
                        pattern=pattern,
                        effect_size=effect,
                        sparsity=sparsity,
                        rng=rng,
                    )

                    model = FlashS(
                        n_features=500,
                        n_scales=7,
                        normalize=False,
                        log_transform=False,
                        random_state=rep,
                    )
                    result = model.fit(coords).test(X, gene_names=gene_names)

                    # Compute TPR and FDR
                    is_svg_arr = np.array(is_svg)
                    tested_mask = np.array([g in result.gene_names for g in gene_names])
                    # Map results back to original gene indices
                    name_to_idx = {n: i for i, n in enumerate(result.gene_names)}

                    significant = np.zeros(len(gene_names), dtype=bool)
                    for j, g in enumerate(gene_names):
                        if g in name_to_idx:
                            idx = name_to_idx[g]
                            significant[j] = result.qvalues[idx] < 0.05

                    # SVG genes that were tested and called significant
                    svg_tested = is_svg_arr & tested_mask
                    n_svg_tested = svg_tested.sum()
                    n_tp = (significant & is_svg_arr).sum()
                    n_fp = (significant & ~is_svg_arr).sum()
                    n_sig = significant.sum()

                    tpr = n_tp / max(n_svg_tested, 1)
                    fdr = n_fp / max(n_sig, 1)

                    rows.append({
                        "pattern": pattern,
                        "effect_size": effect,
                        "sparsity": sparsity,
                        "rep": rep,
                        "tpr": tpr,
                        "fdr": fdr,
                        "n_tp": int(n_tp),
                        "n_fp": int(n_fp),
                        "n_sig": int(n_sig),
                        "n_svg_tested": int(n_svg_tested),
                        "n_tested": result.n_tested,
                    })

                # Print summary for this config
                config_rows = [r for r in rows if r["pattern"] == pattern
                               and r["effect_size"] == effect
                               and r["sparsity"] == sparsity]
                mean_tpr = np.mean([r["tpr"] for r in config_rows])
                mean_fdr = np.mean([r["fdr"] for r in config_rows])
                print(f"    Mean TPR={mean_tpr:.3f}, Mean FDR={mean_fdr:.3f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "results", "power_analysis.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    return df


# ---------------------------------------------------------------------------
# Experiment 3: Ablation over D (n_features)
# ---------------------------------------------------------------------------

def run_ablation_D():
    """
    Ablation study over D (n_features).

    D in [50, 100, 200, 500, 1000], n_scales=7.
    200 SVG + 200 null genes, hotspot pattern, effect=0.3.
    5 reps each.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Ablation over D (n_features)")
    print("=" * 70)

    grid_size = 50
    n_svg = 200
    n_null = 200
    D_values = [50, 100, 200, 500, 1000]
    n_scales = 7
    n_reps = 5
    rows = []

    coords = make_grid(grid_size)

    for D in D_values:
        print(f"\n  D={D}, n_scales={n_scales}")

        for rep in range(n_reps):
            rng = np.random.default_rng(42 + rep + D * 100)

            X, gene_names, is_svg = simulate_svg_data(
                grid_size=grid_size,
                n_svg=n_svg,
                n_null=n_null,
                pattern="hotspot",
                effect_size=0.3,
                sparsity=0.3,
                rng=rng,
            )

            model = FlashS(
                n_features=D,
                n_scales=n_scales,
                normalize=False,
                log_transform=False,
                random_state=rep,
            )

            t0 = time.time()
            result = model.fit(coords).test(X, gene_names=gene_names)
            runtime = time.time() - t0

            # Compute TPR and FDR
            is_svg_arr = np.array(is_svg)
            tested_mask = np.array([g in result.gene_names for g in gene_names])
            name_to_idx = {n: i for i, n in enumerate(result.gene_names)}

            significant = np.zeros(len(gene_names), dtype=bool)
            for j, g in enumerate(gene_names):
                if g in name_to_idx:
                    idx = name_to_idx[g]
                    significant[j] = result.qvalues[idx] < 0.05

            svg_tested = is_svg_arr & tested_mask
            n_svg_tested = svg_tested.sum()
            n_tp = (significant & is_svg_arr).sum()
            n_fp = (significant & ~is_svg_arr).sum()
            n_sig = significant.sum()

            tpr = n_tp / max(n_svg_tested, 1)
            fdr = n_fp / max(n_sig, 1)

            rows.append({
                "D": D,
                "n_scales": n_scales,
                "rep": rep,
                "tpr": tpr,
                "fdr": fdr,
                "runtime_s": runtime,
                "n_tested": result.n_tested,
            })

            print(f"    Rep {rep}: TPR={tpr:.3f}, FDR={fdr:.3f}, time={runtime:.2f}s")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "results", "ablation_D.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Summary
    print("\n  Summary by D:")
    for D in D_values:
        sub = df[df["D"] == D]
        print(f"    D={D:5d}: TPR={sub['tpr'].mean():.3f}+/-{sub['tpr'].std():.3f}, "
              f"FDR={sub['fdr'].mean():.3f}+/-{sub['fdr'].std():.3f}, "
              f"time={sub['runtime_s'].mean():.2f}s")

    return df


# ---------------------------------------------------------------------------
# Experiment 4: Ablation over n_scales
# ---------------------------------------------------------------------------

def run_ablation_nscales():
    """
    Ablation study over n_scales.

    n_scales in [1, 3, 5, 7, 10], D=500.
    200 SVG + 200 null genes, hotspot pattern, effect=0.3.
    5 reps each.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Ablation over n_scales")
    print("=" * 70)

    grid_size = 50
    n_svg = 200
    n_null = 200
    nscales_values = [1, 3, 5, 7, 10]
    D = 500
    n_reps = 5
    rows = []

    coords = make_grid(grid_size)

    for ns in nscales_values:
        print(f"\n  n_scales={ns}, D={D}")

        for rep in range(n_reps):
            rng = np.random.default_rng(42 + rep + ns * 100)

            X, gene_names, is_svg = simulate_svg_data(
                grid_size=grid_size,
                n_svg=n_svg,
                n_null=n_null,
                pattern="hotspot",
                effect_size=0.3,
                sparsity=0.3,
                rng=rng,
            )

            model = FlashS(
                n_features=D,
                n_scales=ns,
                normalize=False,
                log_transform=False,
                random_state=rep,
            )

            t0 = time.time()
            result = model.fit(coords).test(X, gene_names=gene_names)
            runtime = time.time() - t0

            # Compute TPR and FDR
            is_svg_arr = np.array(is_svg)
            tested_mask = np.array([g in result.gene_names for g in gene_names])
            name_to_idx = {n: i for i, n in enumerate(result.gene_names)}

            significant = np.zeros(len(gene_names), dtype=bool)
            for j, g in enumerate(gene_names):
                if g in name_to_idx:
                    idx = name_to_idx[g]
                    significant[j] = result.qvalues[idx] < 0.05

            svg_tested = is_svg_arr & tested_mask
            n_svg_tested = svg_tested.sum()
            n_tp = (significant & is_svg_arr).sum()
            n_fp = (significant & ~is_svg_arr).sum()
            n_sig = significant.sum()

            tpr = n_tp / max(n_svg_tested, 1)
            fdr = n_fp / max(n_sig, 1)

            rows.append({
                "n_scales": ns,
                "D": D,
                "rep": rep,
                "tpr": tpr,
                "fdr": fdr,
                "runtime_s": runtime,
                "n_tested": result.n_tested,
            })

            print(f"    Rep {rep}: TPR={tpr:.3f}, FDR={fdr:.3f}, time={runtime:.2f}s")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "results", "ablation_nscales.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Summary
    print("\n  Summary by n_scales:")
    for ns in nscales_values:
        sub = df[df["n_scales"] == ns]
        print(f"    n_scales={ns:2d}: TPR={sub['tpr'].mean():.3f}+/-{sub['tpr'].std():.3f}, "
              f"FDR={sub['fdr'].mean():.3f}+/-{sub['fdr'].std():.3f}, "
              f"time={sub['runtime_s'].mean():.2f}s")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FlashS Benchmark Suite")
    print("Post Satterthwaite variance fix (per-scale ||Cov_z||_F^2)")
    print("=" * 70)

    t_total = time.time()

    # Run all experiments
    df_type1 = run_type1_error()
    df_power = run_power_analysis()
    df_ablation_D = run_ablation_D()
    df_ablation_ns = run_ablation_nscales()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to benchmarks/results/")
