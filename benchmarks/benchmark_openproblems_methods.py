"""Open Problems SVG benchmark for multiple methods.

Runs scBSP, Hotspot, and nnSVG on the 50 Open Problems SVG benchmark datasets
and computes Kendall tau correlation against ground truth rankings.
Also re-runs FlashS for direct comparison in the same environment.

Each method produces a gene-level score; higher = more spatially variable.
Kendall tau is computed per feature class then averaged per dataset.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import glob
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

BASE = os.path.expanduser("~/Research/task_spatially_variable_genes/benchmark_data")


def discover_datasets(base=BASE):
    # type: () -> List[str]
    dataset_dirs = sorted(glob.glob(os.path.join(base, "*/*/*")))
    dataset_dirs += sorted(glob.glob(os.path.join(base, "*/*/*/*")))
    seen = set()
    valid = []
    for d in dataset_dirs:
        if d in seen:
            continue
        if (os.path.isfile(os.path.join(d, "dataset.h5ad")) and
                os.path.isfile(os.path.join(d, "solution.h5ad"))):
            seen.add(d)
            valid.append(d)
    return valid


# ---------------------------------------------------------------------------
# Kendall tau evaluation (same as verify_benchmark.py)
# ---------------------------------------------------------------------------

def evaluate_scores(pred_scores, gene_names, sol_var):
    # type: (np.ndarray, list, pd.DataFrame) -> float
    """Compute mean Kendall tau against ground truth.

    Args:
        pred_scores: (n_genes,) array of predicted spatial variability scores
        gene_names: list of gene identifiers (aligned with pred_scores)
        sol_var: DataFrame with columns feature_id, true_spatial_var_score,
                 orig_feature_name

    Returns:
        Mean Kendall tau across feature classes.
    """
    gene_to_score = dict(zip(gene_names, pred_scores))
    all_genes = list(sol_var["feature_id"])
    pred_list = [gene_to_score.get(g, 0.0) for g in all_genes]

    pred_df = pd.DataFrame({
        "feature_id": all_genes,
        "pred_spatial_var_score": pred_list,
    })
    true_df = sol_var[["feature_id", "true_spatial_var_score",
                       "orig_feature_name"]].copy()
    merged = pred_df.merge(true_df, on="feature_id")

    corrs = []
    for _, group in merged.groupby("orig_feature_name", observed=True):
        if len(group) > 1:
            tau, _ = kendalltau(
                group["pred_spatial_var_score"],
                group["true_spatial_var_score"],
            )
            if not np.isnan(tau):
                corrs.append(tau)

    return float(np.mean(corrs)) if corrs else float("nan")


# ---------------------------------------------------------------------------
# Method runners: each returns (scores, gene_names) tuple
# ---------------------------------------------------------------------------

def run_flashs(coords, X, gene_names):
    # type: (np.ndarray, ..., list) -> np.ndarray
    """FlashS SOTA configuration."""
    from flashs import FlashS
    model = FlashS(adjustment="none", random_state=42)
    result = model.fit_test(coords, X, gene_names=gene_names)

    es = result.effect_size
    logp = -np.log10(np.clip(result.pvalues, 1e-300, 1))
    es_norm = (es - es.min()) / (es.max() - es.min() + 1e-10)
    logp_norm = (logp - logp.min()) / (logp.max() - logp.min() + 1e-10)
    scores = es_norm + logp_norm

    return dict(zip(result.gene_names, scores))


def run_hotspot(coords, X, gene_names):
    # type: (np.ndarray, ..., list) -> dict
    """Hotspot autocorrelation test."""
    import hotspot

    adata = ad.AnnData(X=X.copy())
    adata.var_names = gene_names
    adata.obsm["spatial"] = coords.copy()
    adata.obs["total_counts"] = np.array(X.sum(axis=1)).flatten()

    hs = hotspot.Hotspot(adata, model="danb", latent_obsm_key="spatial")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    hs_results = hs.compute_autocorrelations()

    # Score: -log10(p) * sign(Z), or just Z-score magnitude
    # Use -log10(FDR) as score (higher = more significant)
    logp = -np.log10(np.clip(hs_results["FDR"].values, 1e-300, 1))
    z = hs_results["Z"].values
    # Combine: genes with high Z and low p are most spatially variable
    scores = logp * np.sign(np.clip(z, 0, None))

    return dict(zip(hs_results.index.tolist(), scores))


def run_scbsp(coords, X, gene_names):
    # type: (np.ndarray, ..., list) -> dict
    """scBSP granularity-based spatial test."""
    from scbsp import granp

    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    result_df = granp(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(X_dense, dtype=np.float64),
    )

    # granp returns DataFrame with columns: gene_names, p_values
    pvals = result_df["p_values"].values
    scores = -np.log10(np.clip(pvals, 1e-300, 1))

    return dict(zip(gene_names, scores))


def run_pretsa(coords, X, gene_names):
    # type: (np.ndarray, ..., list) -> dict
    """PreTSA B-spline F-test for SVG detection."""
    from pretsa_spatial import spatialTest

    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    # PreTSA expects library-size-normalized and log-transformed data
    # Normalize: divide by total counts per cell, multiply by median total
    total_counts = X_dense.sum(axis=1, keepdims=True)
    total_counts[total_counts == 0] = 1
    median_total = np.median(total_counts[total_counts > 0])
    X_norm = X_dense / total_counts * median_total
    X_log = np.log1p(X_norm)

    # PreTSA expects (genes x cells) matrix
    expr = X_log.T  # (n_genes, n_cells)

    coord = {
        "row": coords[:, 0].astype(np.float64),
        "col": coords[:, 1].astype(np.float64),
    }

    result = spatialTest(expr, coord, knot=0)
    # result columns: logpval, pval, fstat
    pvals = result[:, 1]
    scores = -np.log10(np.clip(pvals, 1e-300, 1))

    return dict(zip(gene_names, scores))


def run_nnsvg(coords, X, gene_names):
    # type: (np.ndarray, ..., list) -> dict
    """nnSVG via rpy2."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    n_cells, n_genes = X_dense.shape

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.r('''
            library(nnSVG)
            library(SpatialExperiment)
        ''')

        # Create count matrix (genes x cells) for R
        count_mat = X_dense.T.astype(np.float64)
        r_counts = ro.r['matrix'](count_mat, nrow=n_genes, ncol=n_cells)

        # Create spatial coords
        r_coords = ro.r['data.frame'](
            x=ro.FloatVector(coords[:, 0].astype(np.float64)),
            y=ro.FloatVector(coords[:, 1].astype(np.float64)),
        )

        ro.globalenv['counts_mat'] = r_counts
        ro.globalenv['coords_df'] = r_coords
        ro.globalenv['gene_names'] = ro.StrVector(gene_names)
        ro.globalenv['cell_names'] = ro.StrVector(
            ["cell_%d" % i for i in range(n_cells)])

        ro.r('''
            rownames(counts_mat) <- gene_names
            colnames(counts_mat) <- cell_names
            rownames(coords_df) <- cell_names

            # Create SpatialExperiment
            spe <- SpatialExperiment(
                assays = list(counts = counts_mat),
                spatialCoords = as.matrix(coords_df)
            )

            # Filter low-expressed genes
            spe <- filter_genes(spe, filter_genes_ncounts = 3,
                               filter_genes_pcspots = 0.5)

            # Log-transform
            spe <- logNormCounts(spe)

            # Run nnSVG
            spe <- nnSVG(spe, verbose = FALSE)

            # Extract results
            nnsvg_res <- rowData(spe)
            nnsvg_genes <- rownames(spe)
            nnsvg_pvals <- nnsvg_res$padj
            nnsvg_lfc <- nnsvg_res$LR_stat
        ''')

        r_genes = list(ro.globalenv['nnsvg_genes'])
        r_pvals = np.array(ro.globalenv['nnsvg_pvals'])
        r_lfc = np.array(ro.globalenv['nnsvg_lfc'])

    # Score: -log10(padj) (higher = more significant)
    scores = -np.log10(np.clip(r_pvals, 1e-300, 1))

    return dict(zip(r_genes, scores))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "FlashS": run_flashs,
    "PreTSA": run_pretsa,
    "Hotspot": run_hotspot,
    "scBSP": run_scbsp,
    "nnSVG": run_nnsvg,
}


def check_methods():
    # type: () -> List[str]
    available = []
    checks = {
        "FlashS": lambda: __import__("flashs"),
        "PreTSA": lambda: __import__("pretsa_spatial"),
        "Hotspot": lambda: (__import__("hotspot"), __import__("anndata")),
        "scBSP": lambda: __import__("scbsp"),
        "nnSVG": lambda: (
            __import__("rpy2.robjects"),
            __import__("rpy2").robjects.r('library(nnSVG)'),
        ),
    }
    for name in METHOD_REGISTRY:
        try:
            checks[name]()
            available.append(name)
            print("  [OK] %s" % name)
        except Exception as e:
            print("  [SKIP] %s — %s" % (name, str(e)[:80]))
    return available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Open Problems SVG Benchmark — Multi-method evaluation")
    print("=" * 90)

    print("\nChecking available methods ...")
    available = check_methods()
    if not available:
        print("No methods available. Exiting.")
        return

    print("\nWill benchmark: %s" % available)

    valid_dirs = discover_datasets()
    print("Found %d datasets\n" % len(valid_dirs))

    # Load existing results for incremental runs
    out_csv = "openproblems_methods.csv"
    done_keys = set()
    existing_rows = []
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        for _, row in existing_df.iterrows():
            rd = row.to_dict()
            if rd.get("status") == "ok":
                existing_rows.append(rd)
                done_keys.add((rd["method"], rd["dataset"]))
        print("Loaded %d cached results\n" % len(existing_rows))

    rows = list(existing_rows)

    header = "%-12s %-50s %7s %7s %8s %8s %s" % (
        "Method", "Dataset", "Spots", "Genes", "Tau", "Time", "Status")
    print(header)
    print("-" * 110)

    for ddir in valid_dirs:
        short_name = ddir.replace(BASE + "/", "")

        for method_name in available:
            key = (method_name, short_name)
            if key in done_keys:
                continue

            try:
                adata = ad.read_h5ad(os.path.join(ddir, "dataset.h5ad"))
                sol = ad.read_h5ad(os.path.join(ddir, "solution.h5ad"))

                coords = adata.obsm["spatial"]
                X = adata.layers["counts"]
                gene_names = list(adata.var_names)

                t0 = time.time()
                method_fn = METHOD_REGISTRY[method_name]
                gene_to_score = method_fn(coords, X, gene_names)
                elapsed = time.time() - t0

                # Build score array aligned with all genes
                all_scores = np.array([
                    gene_to_score.get(g, 0.0) for g in gene_names
                ])

                # Evaluate
                tau = evaluate_scores(all_scores, gene_names, sol.var)

                print("%-12s %-50s %7d %7d %8.4f %7.1fs ok" % (
                    method_name, short_name, adata.n_obs, adata.n_vars,
                    tau, elapsed), flush=True)

                rows.append({
                    "method": method_name,
                    "dataset": short_name,
                    "n_spots": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "kendall_tau": float(tau),
                    "runtime_s": float(elapsed),
                    "status": "ok",
                    "error_msg": "",
                })

            except Exception as e:
                print("%-12s %-50s %7s %7s %8s %7s ERROR %s" % (
                    method_name, short_name, "", "", "", "",
                    str(e)[:60]), flush=True)

                rows.append({
                    "method": method_name,
                    "dataset": short_name,
                    "n_spots": 0,
                    "n_genes": 0,
                    "kendall_tau": float("nan"),
                    "runtime_s": float("nan"),
                    "status": "error",
                    "error_msg": str(e)[:200],
                })

            # Save incrementally
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        gc.collect()

    # ---- Summary ----
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    for method in available:
        mdf = df[(df["method"] == method) & (df["status"] == "ok")]
        if len(mdf) == 0:
            print("%-12s: no successful runs" % method)
            continue
        taus = mdf["kendall_tau"].dropna()
        times = mdf["runtime_s"]
        print("%-12s: n=%d  mean_tau=%.4f  median_tau=%.4f  "
              "min=%.4f  max=%.4f  total_time=%.1fs" % (
                  method, len(taus), taus.mean(), taus.median(),
                  taus.min(), taus.max(), times.sum()))

    # Save JSON summary
    summary = {}
    for method in available:
        mdf = df[(df["method"] == method) & (df["status"] == "ok")]
        taus = mdf["kendall_tau"].dropna()
        if len(taus) > 0:
            summary[method] = {
                "mean_tau": float(taus.mean()),
                "median_tau": float(taus.median()),
                "n_datasets": int(len(taus)),
                "n_errors": int(
                    len(df[(df["method"] == method) & (df["status"] != "ok")])),
            }

    out_json = "openproblems_methods.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults: %s, %s" % (out_csv, out_json))


if __name__ == "__main__":
    main()
