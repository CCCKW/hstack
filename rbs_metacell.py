import os
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import stark as sk
import scanpy as sc
import metacells as mc
import numpy as np
import numpy as np
import pandas as pd
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
from tqdm import tqdm


def main():
    import pandas as pd
    result = pd.DataFrame(columns=['mean_purity', 'acc', 'global_score', 'wcos', 'hwis'])
    pbar = tqdm(range(11,40), desc="MetaCell 数量", unit="num")
    for num in pbar:
        lb = []
        path = '/Users/ckw/warehouse/metacell/data/test_700_snm3c'
        for val in os.listdir(path):
            if val.endswith('.pairs'):
                lb.append(val.split('.pairs')[0].split('_')[1])
        lb = ['ExcNeuron' if x in ['L23', 'L4', 'L5', 'L6'] else x for x in lb]

        pca_vec = np.load('/Users/ckw/warehouse/metacell/stark/test_output/pca_vec_500000.npy')
        umap_vec = np.load('/Users/ckw/warehouse/metacell/stark/test_output/umap_vec_500000.npy')
        cell_embeddings = pca_vec  # 或者 umap_vec，取决于你想用哪个作为输入
        print(pca_vec.shape, umap_vec.shape, len(lb))
        adata = ad.AnnData(cell_embeddings)
        adata.obs['cell_type'] = lb
        adata.obsm['X_pca'] = pca_vec
        adata.obsm['X_umap'] = umap_vec


        # 用PCA建KNN图
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)

        # 之后可以用图分区（leiden/louvain作为对比基线）
        sc.tl.leiden(adata, resolution=0.1)
        # mc.pl.compute_direct_metacells(adata, random_seed=123456)
        # 关键步骤：把PCA坐标转成MC2能接受的格式
        # 1. 平移到非负
        pca_shifted = pca_vec - pca_vec.min()
        # 2. 缩放到合理的UMI量级（模拟UMI counts）
        pca_scaled = (pca_shifted / pca_shifted.max() * 1000).astype(np.float32)

        # 构造AnnData，X用处理后的PCA
        adata = ad.AnnData(X=pca_scaled)
        adata.obs['cell_type'] = lb



        # 给每个PC维度一个名字（当作"基因"）
        import pandas as pd
        adata.var_names = [f"PC{i}" for i in range(pca_scaled.shape[1])]
        adata.obs_names = [f"cell{i}" for i in range(pca_scaled.shape[0])]

        mc.ut.set_name(adata, "hic_data")

        mc.pl.mark_lateral_genes(adata)  # 不传任何基因，仅初始化
        n_cells = adata.n_obs
        target_n_metacells = num  # 你想要的metacell数

        # 反推target_metacell_umis
        total_umi = adata.X.sum()
        target_umis = int(total_umi / target_n_metacells)
        print(f"Total UMI: {total_umi}, Target UMIs per metacell: {target_umis}")
        # 关闭MC2内部的基因筛选（因为PC维度不是真正的基因）
        mc.pl.compute_direct_metacells(
            adata,
            random_seed=123456,
            target_metacell_umis=target_umis,  # 关键：控制metacell数量
            select_min_gene_total=0,
            select_min_gene_top3=0,
            select_min_gene_relative_variance=0,
            select_min_genes=1,
            deviants_min_gene_fold_factor=8,
            deviants_max_cell_fraction=0.25,
            min_metacell_size=5,
        )


        mdata = mc.pl.collect_metacells(adata, random_seed=123456)
        adata.obs.columns = ['label', 'dissolved', 'metacell', 'm_name']
        adata.uns['X_pca'] = pca_vec    
        adata.uns['X_umap'] = umap_vec
        print(mdata)
        hdata = sk.create_hdata_from_adata(adata,
                                 data_dir="/Users/ckw/warehouse/metacell/data/test_700_snm3c",
                                output_dir="/Users/ckw/warehouse/metacell/stark/test_output",
                                genome_reference_path="/Users/ckw/warehouse/metacell/hg19.fa.chrom.sizes",
                                chrom_list=[f"chr{i}" for i in range(1, 23)],
                                resolution=[500000])

        purity_df, metrics = sk.tl.evaluate(hdata, hdata.obs['label'])
        vals = np.array(metrics.values())
        print(vals)
        result.loc[num] = vals
    result.to_csv('mc2.csv')
    
    
    
if __name__ == "__main__":
    main()