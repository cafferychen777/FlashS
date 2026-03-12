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
model = FlashS(adjustment='none', random_state=42)

result = model.fit_test(coords, X, gene_names=list(adata.var_names))

# Use effect size as the default ranking output.
# result arrays are aligned with input gene order (adata.var_names)
scores = result.effect_size

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
