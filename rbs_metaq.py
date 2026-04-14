import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
import anndata as ad
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from MetaQ_sc import run_metaq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import os
import h5py
import matplotlib.pyplot as plt
import stark as sk


def main():
    result = pd.DataFrame(
        columns=[
            "mean_purity",
            "acc",
            "global_score",
            "wcos",
            "hwis",
            "compactness",
            "separation",
        ]
    )

    pbar = tqdm(range(11, 40), desc="MetaCell 数量", unit="num")
    for num in pbar:
        run_metaq(
            data_path=[
                "/Users/ckw/warehouse/metacell/MetaQ-main/500kb.h5ad",
            ],  # the path to the input h5ad data
            train_epoch=300,
            data_type=[
                "ADT",
            ],
            type_key="cell_type",
            # the type of the input data
            metacell_num=num,  # the target number of metacells
            save_name="metaq",  # the file name prefix when saving the results
            device="cpu",
        )

        adata = sc.read_h5ad(f"./save/metaq_{num}metacell_ids.h5ad")
        lb = []
        path = "/Users/ckw/warehouse/metacell/data/test_700_snm3c"
        for val in os.listdir(path):
            if val.endswith(".pairs"):
                lb.append(val.split(".pairs")[0].split("_")[1])
        lb = ["ExcNeuron" if x in ["L23", "L4", "L5", "L6"] else x for x in lb]
        pca_vec = np.load(
            "/Users/ckw/warehouse/metacell/stark/test_output/pca_vec_500000.npy"
        )
        umap_vec = np.load(
            "/Users/ckw/warehouse/metacell/stark/test_output/umap_vec_500000.npy"
        )
        print(pca_vec.shape, umap_vec.shape, len(lb))
        adata.obs["cell_type"] = lb
        adata.obs["label"] = lb
        adata.obsm["X_pca"] = pca_vec
        adata.obsm["X_umap"] = umap_vec
        adata.uns["X_pca"] = pca_vec
        adata.uns["X_umap"] = umap_vec

        hdata = sk.create_hdata_from_adata(
            adata,
            data_dir="/Users/ckw/warehouse/metacell/data/test_700_snm3c",
            output_dir="/Users/ckw/warehouse/metacell/stark/test_output",
            genome_reference_path="/Users/ckw/warehouse/metacell/hg19.fa.chrom.sizes",
            chrom_list=[f"chr{i}" for i in range(1, 23)],
            resolution=[500000],
        )
        purity_df, metrics = sk.tl.evaluate(hdata, hdata.obs["label"])
        # result[num] = metrics
        res_df, metrics_summary = sk.tl.evaluate_metacell(
            hdata=hdata,
            use_view=None,  # 默认选第0个组学视角，也可以传入字典里的特定 key
            metric="euclidean",  # 或 'cosine'
        )

        vals = list(metrics.values())

        print(vals)
        result.loc[num] = np.array(vals)
        # break
    result.to_csv("./benchmark/metaQ.csv")


if __name__ == "__main__":
    main()
