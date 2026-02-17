import warnings
warnings.filterwarnings('ignore')

import anndata as ad
import numpy as np
import pandas as pd

# VIASH START
par = {
    'input_data': 'resources_test/task_spatially_variable_genes/mouse_brain_coronal/dataset.h5ad',
    'output': 'output.h5ad'
}
meta = {
    'name': 'flashs'
}
# VIASH END

print('Load data', flush=True)
adata = ad.read_h5ad(par['input_data'])

print('Run FlashS', flush=True)
from flashs import FlashS

coords = adata.obsm['spatial']
X = adata.layers['counts']

# Use raw counts with multi-kernel Cauchy combination.
# adjustment='none' because ranking relies on the composite score.
model = FlashS(adjustment='none', random_state=42)

result = model.fit_test(coords, X, gene_names=list(adata.var_names))

# Score: effect_size + multi-kernel Cauchy combined p-value
# result arrays are aligned with input gene order (adata.var_names)
es = result.effect_size
logp_multi = -np.log10(np.clip(result.pvalues, 1e-300, 1))

es_norm = (es - es.min()) / (es.max() - es.min() + 1e-10)
multi_norm = (logp_multi - logp_multi.min()) / (logp_multi.max() - logp_multi.min() + 1e-10)
scores = es_norm + multi_norm

df = pd.DataFrame({
    'feature_id': list(adata.var_names),
    'pred_spatial_var_score': scores
})

output = ad.AnnData(
    var=df,
    uns={
        'dataset_id': adata.uns['dataset_id'],
        'method_id': meta['name']
    }
)

print('Write output to file', flush=True)
output.write_h5ad(par['output'])
