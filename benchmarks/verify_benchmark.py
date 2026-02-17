"""Verify Open Problems SVG benchmark result (tau=0.934) after variance fix.

Uses the SOTA configuration from open_problems/script.py:
- FlashS(adjustment='none', random_state=42) with defaults
  (n_features=500, n_scales=7, normalize=False, log_transform=False)
- Score: es_norm + multi_norm
"""

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from flashs import FlashS
import time
import os
import glob
import json

base = os.path.expanduser("~/Research/task_spatially_variable_genes/benchmark_data")

# Find all datasets
dataset_dirs = sorted(glob.glob(os.path.join(base, "*/*/*")))
dataset_dirs += sorted(glob.glob(os.path.join(base, "*/*/*/*")))

seen = set()
valid_dirs = []
for d in dataset_dirs:
    if d in seen:
        continue
    if os.path.isfile(os.path.join(d, "dataset.h5ad")) and os.path.isfile(os.path.join(d, "solution.h5ad")):
        seen.add(d)
        valid_dirs.append(d)

print(f"Found {len(valid_dirs)} datasets")
print()
header = f"{'Dataset':<55} {'Spots':>7} {'Genes':>7} {'Kendall':>8} {'Time':>8}"
print(header)
print("-" * 90)

results = []
errors = []

for ddir in valid_dirs:
    short_name = ddir.replace(base + "/", "")
    data_path = os.path.join(ddir, "dataset.h5ad")
    sol_path = os.path.join(ddir, "solution.h5ad")

    try:
        adata = ad.read_h5ad(data_path)
        sol = ad.read_h5ad(sol_path)

        coords = adata.obsm["spatial"]
        X = adata.layers["counts"]

        t0 = time.time()
        # SOTA configuration (same as open_problems/script.py)
        model = FlashS(adjustment='none', random_state=42)
        result = model.fit_test(coords, X, gene_names=list(adata.var_names))
        elapsed = time.time() - t0

        # SOTA scoring: effect_size + normalized -log10(p)
        es = result.effect_size
        logp = -np.log10(np.clip(result.pvalues, 1e-300, 1))

        es_norm = (es - es.min()) / (es.max() - es.min() + 1e-10)
        logp_norm = (logp - logp.min()) / (logp.max() - logp.min() + 1e-10)
        combined = es_norm + logp_norm

        gene_to_score = dict(zip(result.gene_names, combined))
        pred_scores = [gene_to_score.get(g, 0.0) for g in adata.var_names]

        pred_df = pd.DataFrame({
            "feature_id": list(adata.var_names),
            "pred_spatial_var_score": pred_scores
        })
        true_df = sol.var[["feature_id", "true_spatial_var_score", "orig_feature_name"]].copy()
        merged = pred_df.merge(true_df, on="feature_id")

        corrs = []
        for gname, group in merged.groupby("orig_feature_name", observed=True):
            if len(group) > 1:
                tau, _ = kendalltau(group["pred_spatial_var_score"], group["true_spatial_var_score"])
                if not np.isnan(tau):
                    corrs.append(tau)

        mean_corr = np.mean(corrs) if corrs else float("nan")
        row = f"{short_name:<55} {adata.n_obs:>7} {adata.n_vars:>7} {mean_corr:>8.4f} {elapsed:>7.1f}s"
        print(row, flush=True)
        results.append({
            'dataset': short_name,
            'n_spots': int(adata.n_obs),
            'n_genes': int(adata.n_vars),
            'kendall_tau': float(mean_corr),
            'runtime_s': float(elapsed),
        })

    except Exception as e:
        print(f"{short_name:<55} ERROR: {str(e)[:60]}", flush=True)
        errors.append({'dataset': short_name, 'error': str(e)})

print()
print("=" * 90)
valid_corrs = [r['kendall_tau'] for r in results if not np.isnan(r['kendall_tau'])]
print(f"Datasets tested: {len(results)} / {len(valid_dirs)} (errors: {len(errors)})")
print(f"Mean Kendall tau:   {np.mean(valid_corrs):.4f}")
print(f"Median Kendall tau: {np.median(valid_corrs):.4f}")
print(f"Min:   {np.min(valid_corrs):.4f}")
print(f"Max:   {np.max(valid_corrs):.4f}")
print(f"Total time: {sum(r['runtime_s'] for r in results):.1f}s")

# Save results
out = {
    'mean_tau': float(np.mean(valid_corrs)),
    'median_tau': float(np.median(valid_corrs)),
    'min_tau': float(np.min(valid_corrs)),
    'max_tau': float(np.max(valid_corrs)),
    'n_datasets': len(results),
    'n_errors': len(errors),
    'results': results,
    'errors': errors,
}
with open(os.path.expanduser('~/FlashS/benchmarks/benchmark_verification.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nResults saved to ~/FlashS/benchmarks/benchmark_verification.json")
