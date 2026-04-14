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
            
        # 挂载 Higashi 预处理好的单细胞接触稀疏矩阵
        temp_raw_dir = os.path.join(hdata.output_dir, f'temp_{res}', 'raw')
        if os.path.exists(temp_raw_dir):
            if res not in hdata.views_mat:
                hdata.views_mat[res] = {}
            for chrom in hdata.chrom_list:
                mat_path = os.path.join(temp_raw_dir, f'{chrom}_sparse_adj.npy')
                if os.path.exists(mat_path):
                    hdata.views_mat[res][chrom] = np.load(mat_path, allow_pickle=True)

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


def cal_is(hdata, resolution=50000, window=6, n_jobs=10, force=False):
    """
    提取并计算 hdata 中所有细胞的全局绝缘分数特征矩阵 (Cells x Bins)。
    直接使用通过 Higashi 插补/平滑后挂载在 hdata.views_mat 中的稀疏矩阵进行计算。
    如果本地已有缓存 (.npy)，则直接秒速载入。
    
    参数:
    hdata: HData 对象，需已运行 process_and_load
    resolution: 扫描分辨率，默认 50kb
    window: 滑动窗口 (bin的数量)
    n_jobs: 线程池并发数 (由于纯内存计算，使用线程池即可)
    force: 是否强制不使用本地 npy 缓存而重新解析
    """
    import os
    import numpy as np
    import concurrent.futures
    from tqdm import tqdm
    from stark.utils.tad import compute_insulation_score
    
    output_path = os.path.join(hdata.output_dir, f"is_vec_{resolution}.npy")
    
    # 1. 缓存直接读取
    if os.path.exists(output_path) and not force:
        print(f"✅ 找到已缓存的 IS 单细胞绝缘分矩阵 {output_path}，正在直接挂载 ...")
        hdata.views_is[resolution] = np.load(output_path)
        return hdata.views_is[resolution]
        
    print(f"🚀 开始基于 Higashi 平滑矩阵计算单细胞绝缘分矩阵 (分辨率: {resolution}, Window: {window}, 线程数: {n_jobs})...")
    
    if resolution not in hdata.views_mat or not hdata.views_mat[resolution]:
        print(f"❌ 警告: hdata.views_mat 中没有找到 {resolution} 对应的接触矩阵！请先运行 process_and_load。")
        return None
        
    chrom_list = hdata.chrom_list
    mat_dict = hdata.views_mat[resolution]
    
    # 检查是否所有 chromosome 都准备好了
    missing_chroms = [c for c in chrom_list if c not in mat_dict]
    if missing_chroms:
        print(f"❌ 警告: 缺少以下染色体的接触矩阵: {missing_chroms}")
        return None
        
    cell_num = len(mat_dict[chrom_list[0]])
    
    def _compute_cell_is(cell_idx):
        cell_is_list = []
        for chrom in chrom_list:
            mat = mat_dict[chrom][cell_idx]
            chrom_is = compute_insulation_score(mat, window=window, normalize=True)
            cell_is_list.append(chrom_is)
        return np.concatenate(cell_is_list)
        
    results = []
    # 由于字典对象跨进程传递极大增加 IPC 消耗，我们使用 ThreadPoolExecutor 
    # compute_insulation_score 里面的 Numpy 高强度计算大多会释放 GIL
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for res_is in tqdm(executor.map(_compute_cell_is, range(cell_num)), total=cell_num, desc="Calculating IS"):
            results.append(res_is)
            
    is_matrix = np.vstack(results)
    
    np.save(output_path, is_matrix)
    hdata.views_is[resolution] = is_matrix
    print(f"🎯 绝缘分矩阵计算完成!")
    print(f"   矩阵尺寸: {is_matrix.shape[0]} 细胞 x {is_matrix.shape[1]} Bins")
    print(f"   已缓存至: {output_path}")
    print(f"   挂载完毕，使用 hdata.views_is[{resolution}] 来调用该结果进行后续特征抽取。")
    
    return is_matrix