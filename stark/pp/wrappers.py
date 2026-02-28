import os
import pandas as pd
import numpy as np
import seaborn as sns

# 导入您原本原封不动的核心处理函数
# 假设当前文件在 stark/pp/ 目录下，pp_code.py 也在同级目录
from .pp_code import stark_process 

def process_and_load(hdata, force_process=True, cpu_num=10, gpu_num=8, scaler_data=True):
    """
    预处理模块的 API 入口 (对标 Scanpy 的 sc.pp.xxx)
    
    1. 调用底层的 stark_process 进行计算
    2. 将计算输出的结果 (PCA, UMAP, Depth 等) 规范地挂载到 HData 对象中
    """
    # ---------------------------------------------------------
    # 步骤 1: 调用您原始的底层计算逻辑，原汁原味，没有任何修改
    # ---------------------------------------------------------
    if force_process:
        stark_process(
            output_dir=hdata.output_dir,
            data_dir=hdata.data_dir,
            genome_reference_path=hdata.genome_reference_path,
            chrom_list=hdata.chrom_list,
            resolution=hdata.resolutions,
            scaler_data=scaler_data,
            cpu_num=cpu_num,
            gpu_num=gpu_num
        )
        print("✅ 数据底层处理 (stark_process) 完成。")

    # ---------------------------------------------------------
    # 步骤 2: 将生成的结果矩阵挂载到 hdata.views_* 中
    # ---------------------------------------------------------
    for res in hdata.resolutions:
        pca_path = os.path.join(hdata.output_dir, f'pca_vec_{res}.npy')
        umap_path = os.path.join(hdata.output_dir, f'umap_vec_{res}.npy')
        emb_path = os.path.join(hdata.output_dir, f'embedding_vec_{res}.npy')
        
        # 只要文件存在，就自动装载进 HData
        if os.path.exists(pca_path):
            hdata.views_pca[res] = np.load(pca_path)
        if os.path.exists(umap_path):
            hdata.views_umap[res] = np.load(umap_path)
        if os.path.exists(emb_path):
            hdata.views_embedding[res] = np.load(emb_path)

    # ---------------------------------------------------------
    # 步骤 3: 提取深度 (Depth) 和真实细胞标签 (Label)，放入 hdata.obs
    # ---------------------------------------------------------
    # 读取深度
    depth_path = os.path.join(hdata.output_dir, "depth.txt")
    if os.path.exists(depth_path):
        hdata.obs['depth'] = pd.read_csv(depth_path, header=None)[0].values

    # # 提取标签（完全复用您原代码 os.listdir 的遍历逻辑，确保和 depth.txt 顺序绝对一致）
    # labels = []
    # for pair in os.listdir(hdata.data_dir):
    #     if pair.endswith(".pairs.gz") or pair.endswith(".pairs"):
    #         # 取出形如 _Astro_ 的中间部分作为 Label
    #         label = pair.split('.pairs')[0].split('_')[1] 
    #         labels.append(label)
            
    # hdata.obs['label'] = labels

    # ---------------------------------------------------------
    # 步骤 4: 生成图例颜色映射，存入非结构化字典 hdata.uns 中
    # ---------------------------------------------------------
    # unique_labels = list(set(labels))
    # palette = sns.color_palette("husl", len(unique_labels))
    # label_color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    # hdata.uns['label_colors'] = label_color_map
    
    print("✅ 数据已成功挂载到 HData 对象中。")