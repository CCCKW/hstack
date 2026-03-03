import SEACells
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
import anndata as ad
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import os
import h5py
import stark as sk
def main():
    result = {}

    pbar = tqdm(range(15,75), desc="MetaCell 数量", unit="num")
    for num in pbar:
        

        lb = []
        path = '/Users/ckw/warehouse/metacell/data/test_700_snm3c'
        for val in os.listdir(path):
            if val.endswith('.pairs'):
                lb.append(val.split('.pairs')[0].split('_')[1])
        pca_vec = np.load('/Users/ckw/warehouse/metacell/stark/test_output/pca_vec_500000.npy')
        umap_vec = np.load('/Users/ckw/warehouse/metacell/stark/test_output/umap_vec_500000.npy')
        cell_embeddings = pca_vec  # 或者 umap_vec，取决于你想用哪个作为输入
        print(pca_vec.shape, umap_vec.shape, len(lb))
        adata = ad.AnnData(cell_embeddings)
        adata.obs['cell_type'] = lb
        adata.obsm['X_pca'] = pca_vec
        adata.obsm['X_umap'] = umap_vec

        n_SEACells = num
        build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
                                # This would be replaced by 'X_svd' for ATAC data

        ## Additional parameters
        n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells

        model = SEACells.core.SEACells(adata, 
                    build_kernel_on=build_kernel_on, 
                    n_SEACells=n_SEACells, 
                    n_waypoint_eigs=n_waypoint_eigs,
                    convergence_epsilon = 1e-5,
                        use_gpu =False)

        model.construct_kernel_matrix()
        M = model.kernel_matrix
        model.initialize_archetypes()
        model.fit(min_iter=10, max_iter=200)
        
        adata.obs.columns = ['label','SEAcell', 'metacell']
        adata.uns['X_pca'] = pca_vec    
        adata.uns['X_umap'] = umap_vec

        hdata = sk.create_hdata_from_adata(adata,
                                    data_dir="/Users/ckw/warehouse/metacell/data/test_700_snm3c",
                                output_dir="/Users/ckw/warehouse/metacell/stark/test_output",
                                genome_reference_path="/Users/ckw/warehouse/metacell/hg19.fa.chrom.sizes",
                                chrom_list=[f"chr{i}" for i in range(1, 23)],
                                resolution=500000)
        purity_df, metrics = sk.tl.evaluate(hdata, hdata.obs['label'])
        result[num] = metrics
    np.save('rbs_seacell.npy', result)
    

if __name__ == "__main__":
    main()