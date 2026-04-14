from .Higashi_wrapper import *
import os
import shutil
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns
from sklearn.preprocessing import normalize
import concurrent.futures
import gzip

def count_valid_lines(file_path):
    """
    高效计算文件的有效行数（跳过以 '#' 开头的注释行）。
    使用生成器按行读取，极大地节省内存，替代耗时的 pd.read_csv。
    """
    count = 0
    # 根据后缀判断是否需要 gzip 解压读取
    if file_path.endswith('.gz'):
        opener = gzip.open(file_path, 'rt')
    else:
        opener = open(file_path, 'r')
        
    with opener as f:
        for line in f:
            if not line.startswith('#'):
                count += 1
    return count

def load_data(file_path, n_chrom=19, n_components=0.75, random_state=42, scaler_data=True):
    data = h5py.File(file_path, 'r')
    data  = [data['cell'][str(i)][:] for i in range(n_chrom)]
    data = np.concatenate(data, axis=1)
    
    if scaler_data:
        scaler = StandardScaler()
        embedding = scaler.fit_transform(data)
    else:
        embedding = data
    pca= PCA(n_components=n_components, random_state=random_state)
    pca_vec = pca.fit_transform(embedding)
    umap_model = UMAP(random_state=random_state,min_dist=1)
    umap_vec  = umap_model.fit_transform(pca_vec)
    if np.min(embedding) < 0:
        embedding += np.abs(np.min(embedding))
        
    # pca_vec = normalize(pca_vec, norm='l2', axis=1)
    
    return pca_vec, umap_vec, embedding

def stark_process(output_dir,
                  data_dir,
                  genome_reference_path,
                  chrom_list,
                  resolution,
                  scaler_data=True,
                  cpu_num=10,
                  gpu_num=8):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('====== calculating depth =======')
    # depth
    depth_output_path = os.path.join(output_dir, "depth.txt")
    
    # 1. 严格按照 os.listdir 的顺序收集文件路径
    files_to_process = []
    for pair in os.listdir(data_dir):
        if pair.endswith(".pairs.gz") or pair.endswith(".pairs"):
            pair_path = os.path.join(data_dir, pair)
            files_to_process.append(pair_path)
            
    # 2. 并行计算行数，executor.map 会自动保证输出结果列表与输入列表顺序严格一致
    if not os.path.exists(depth_output_path):
        depth = []
        if files_to_process:
            # 使用你传入的 cpu_num 控制并发数
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
                depth = list(executor.map(count_valid_lines, files_to_process))
                
        # 3. 写入文件
        with open(depth_output_path, "w") as f:
            for d in depth:
                f.write(f"{d}\n")
    else:
        print("Depth file already exists. Skipping depth calculation.")
        depth = pd.read_csv(depth_output_path, header=None)[0].values.tolist()
            
    print('====== depth calculation completed =======')
    
    
    if isinstance(resolution, list):
        print('====== Processing multiple resolutions =======')
        for i in range(len(resolution)):
            temp_res = resolution[i]
            print('====== Processing resolution: {} ======='.format(temp_res))
            if os.path.exists(output_dir + f"/pca_vec_{temp_res}.npy") and os.path.exists(output_dir + f"/umap_vec_{temp_res}.npy") and os.path.exists(output_dir + f"/embedding_vec_{temp_res}.npy"):
                print(f"Files for resolution {temp_res} already exist. Skipping processing.")
                continue
          
            
            temp_dir = os.path.join(output_dir, f"temp_{temp_res}")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
                
            config = {
                "data_dir": data_dir,
                    "temp_dir": temp_dir,
                    "structured": True,
                    "input_format": "higashi_v2",
                    "header_included": True,
                    "genome_reference_path": genome_reference_path,
                    "chrom_list": chrom_list,
                    "resolution": temp_res,
                    "resolution_cell": temp_res,
                    "resolution_fh": [temp_res],
                    "minimum_distance": temp_res,
                    "maximum_distance": -1,
                    "local_transfer_range": 0,
                    "dimensions": 128,
                    "cpu_num": cpu_num,
                    "gpu_num": gpu_num,
                    
            }
            
            
            model = Higashi(config)
            model.process_data()

            if isinstance(scaler_data, list):
                
            
                pca_vec, umap_vec, embedding_vec = load_data(os.path.join(temp_dir, "node_feats.hdf5"),scaler_data=scaler_data[i])
            else:
                pca_vec, umap_vec, embedding_vec = load_data(os.path.join(temp_dir, "node_feats.hdf5"),scaler_data=scaler_data)
            
            # save
            print('====== Saving PCA, UMAP, and embedding vectors =======')
            pca_output_path = os.path.join(output_dir, f"pca_vec_{temp_res}.npy")
            umap_output_path = os.path.join(output_dir, f"umap_vec_{temp_res}.npy")
            embedding_output_path = os.path.join(output_dir, f"embedding_vec_{temp_res}.npy")
            
            np.save(pca_output_path, pca_vec)
            np.save(umap_output_path, umap_vec)
            np.save(embedding_output_path, embedding_vec)
            print('====== Saving completed =======')
            
    else:
        
        temp_dir = os.path.join(output_dir, f"temp_{resolution}")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        config = {
            "data_dir": data_dir,
                "temp_dir": temp_dir,
                "structured": True,
                "input_format": "higashi_v2",
                "header_included": True,
                "genome_reference_path": genome_reference_path,
                "chrom_list": chrom_list,
                "resolution": resolution,
                "resolution_cell": resolution,
                "resolution_fh": [resolution],
                "minimum_distance": resolution,
                "maximum_distance": -1,
                "local_transfer_range": 0,
                "dimensions": 128,
                "cpu_num": cpu_num,
                "gpu_num": gpu_num,
                
        }
        
        
        model = Higashi(config)
        model.process_data()

        
        
        pca_vec, umap_vec, embedding_vec = load_data(os.path.join(temp_dir, "node_feats.hdf5"),scaler_data=scaler_data)
        
        # save
        print('====== Saving PCA, UMAP, and embedding vectors =======')
        pca_output_path = os.path.join(output_dir, f"pca_vec_{resolution}.npy")
        umap_output_path = os.path.join(output_dir, f"umap_vec_{resolution}.npy")
        embedding_output_path = os.path.join(output_dir, f"embedding_vec_{resolution}.npy")
        
        np.save(pca_output_path, pca_vec)
        np.save(umap_output_path, umap_vec)
        np.save(embedding_output_path, embedding_vec)
        print('====== Saving completed =======')

if __name__ == "__main__":

    output_dir = "/Users/ckw/warehouse/metacell/v1/data/1mb"
    data_dir = "/Users/ckw/warehouse/metacell/data/test_700_snm3c"
    genome_reference_path = "/Users/ckw/warehouse/metacell/hg19.fa.chrom.sizes"
    chrom_list = [f"chr{i}" for i in range(1, 23)] 
    resolution = 1000000
    stark_process(output_dir,
                    data_dir,
                    genome_reference_path,
                    chrom_list,
                    resolution)
    
    print("This is v1/code.py")