import os
import pandas as pd
import numpy as np
import seaborn as sns

from .pp_code import stark_process
from stark.core.hdata import _make_view_key, MODALITY_HIC, MODALITY_RNA, MODALITY_METH, MODALITY_ATAC

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
    # 步骤 2: 将生成的结果矩阵挂载到 hdata.views_* 中，键格式为 "hic_<resolution>"
    # ---------------------------------------------------------
    for res in hdata.resolutions:
        vk = _make_view_key(MODALITY_HIC, res)

        pca_path = os.path.join(hdata.output_dir, f'pca_vec_{res}.npy')
        umap_path = os.path.join(hdata.output_dir, f'umap_vec_{res}.npy')
        emb_path  = os.path.join(hdata.output_dir, f'embedding_vec_{res}.npy')

        if os.path.exists(pca_path):
            hdata.views_pca[vk] = np.load(pca_path)
        if os.path.exists(umap_path):
            hdata.views_umap[vk] = np.load(umap_path)
        if os.path.exists(emb_path):
            hdata.views_embedding[vk] = np.load(emb_path)

        # 确保 view_config 已注册
        if vk not in hdata.view_configs:
            hdata.add_view_config(vk, modality=MODALITY_HIC, resolution=res,
                                  data_dir=hdata.data_dir)

        # 挂载 Higashi 预处理好的单细胞接触稀疏矩阵
        temp_raw_dir = os.path.join(hdata.output_dir, f'temp_{res}', 'raw')
        if os.path.exists(temp_raw_dir):
            if vk not in hdata.views_mat:
                hdata.views_mat[vk] = {}
            for chrom in hdata.chrom_list:
                mat_path = os.path.join(temp_raw_dir, f'{chrom}_sparse_adj.npy')
                if os.path.exists(mat_path):
                    hdata.views_mat[vk][chrom] = np.load(mat_path, allow_pickle=True)

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


def recompute_embedding(
    hdata,
    resolutions=None,
    n_components=0.75,
    random_state=42,
    scaler_data=True,
    umap_min_dist=1.0,
    umap_n_neighbors=15,
    n_chrom=None,
):
    """
    从已有的 node_feats.hdf5 重新计算 PCA 和 UMAP，并更新 hdata 中对应视图。

    可在 process_and_load 之后调用，无需重新运行 Higashi 处理，只重跑降维步骤。

    参数
    ----
    hdata         : HData 对象
    resolutions   : 要重算的分辨率列表，默认为 hdata.resolutions 全部
    n_components  : PCA 保留方差比例或维数，默认 0.75
    random_state  : 随机种子，默认 42
    scaler_data   : 是否对 node_feats 做 StandardScaler，默认 True
    umap_min_dist : UMAP min_dist 参数，默认 1.0
    umap_n_neighbors : UMAP n_neighbors 参数，默认 15
    n_chrom       : node_feats.hdf5 中读取的染色体数量，None 则自动从文件推断
    """
    import h5py
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from umap import UMAP
    from stark.core.hdata import _make_view_key, MODALITY_HIC

    if resolutions is None:
        resolutions = hdata.resolutions

    for res in resolutions:
        vk = _make_view_key(MODALITY_HIC, res)
        node_feats_path = os.path.join(hdata.output_dir, f"temp_{res}", "node_feats.hdf5")

        if not os.path.exists(node_feats_path):
            print(f"⚠️  找不到 {node_feats_path}，跳过分辨率 {res}。")
            continue

        print(f"🔄 重新计算 PCA/UMAP: 分辨率={res} ...")

        # 读取 node_feats
        with h5py.File(node_feats_path, 'r') as f:
            n = n_chrom if n_chrom is not None else len(f['cell'])
            data = np.concatenate([f['cell'][str(i)][:] for i in range(n)], axis=1)

        if scaler_data:
            embedding = StandardScaler().fit_transform(data)
        else:
            embedding = data

        # 修正负值（保持与原 load_data 一致）
        if np.min(embedding) < 0:
            embedding_for_save = embedding + np.abs(np.min(embedding))
        else:
            embedding_for_save = embedding

        pca_vec = PCA(n_components=n_components, random_state=random_state).fit_transform(embedding)
        umap_vec = UMAP(
            random_state=random_state,
            min_dist=umap_min_dist,
            n_neighbors=umap_n_neighbors,
        ).fit_transform(pca_vec)

        # 保存到磁盘（覆盖旧文件）
        np.save(os.path.join(hdata.output_dir, f"pca_vec_{res}.npy"), pca_vec)
        np.save(os.path.join(hdata.output_dir, f"umap_vec_{res}.npy"), umap_vec)
        np.save(os.path.join(hdata.output_dir, f"embedding_vec_{res}.npy"), embedding_for_save)

        # 更新 hdata 内存中的视图
        hdata.views_pca[vk] = pca_vec
        hdata.views_umap[vk] = umap_vec
        hdata.views_embedding[vk] = embedding_for_save

        print(f"  ✅ 分辨率 {res}: PCA shape={pca_vec.shape}, UMAP shape={umap_vec.shape}")

    print("✅ recompute_embedding 完成。")


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
    
    vk = _make_view_key(MODALITY_HIC, resolution)
    output_path = os.path.join(hdata.output_dir, f"is_vec_{resolution}.npy")

    if os.path.exists(output_path) and not force:
        print(f"✅ 找到缓存 IS 矩阵，直接挂载到 views_is['{vk}']...")
        hdata.views_is[vk] = np.load(output_path)
        return hdata.views_is[vk]

    print(f"🚀 计算 IS 矩阵 (view={vk}, window={window}, threads={n_jobs})...")

    if vk not in hdata.views_mat or not hdata.views_mat[vk]:
        print(f"❌ hdata.views_mat 中未找到 {vk}，请先运行 process_and_load。")
        return None

    chrom_list = hdata.chrom_list
    mat_dict   = hdata.views_mat[vk]

    missing = [c for c in chrom_list if c not in mat_dict]
    if missing:
        print(f"❌ 缺少染色体接触矩阵: {missing}")
        return None

    cell_num = len(mat_dict[chrom_list[0]])

    def _compute_cell_is(cell_idx):
        return np.concatenate([
            compute_insulation_score(mat_dict[ch][cell_idx], window=window, normalize=True)
            for ch in chrom_list
        ])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for res_is in tqdm(executor.map(_compute_cell_is, range(cell_num)),
                           total=cell_num, desc="Calculating IS"):
            results.append(res_is)

    is_matrix = np.vstack(results)
    np.save(output_path, is_matrix)
    hdata.views_is[vk] = is_matrix
    print(f"🎯 IS 矩阵完成: {is_matrix.shape[0]} 细胞 x {is_matrix.shape[1]} Bins，缓存至 {output_path}。")
    return is_matrix


def add_omics_view(hdata, view_key: str, modality: str,
                   pca=None, umap=None, embedding=None,
                   adata=None, obsm_pca_key='X_pca', obsm_umap_key='X_umap',
                   resolution: int = None, data_dir: str = None,
                   n_components: float = 0.75, random_state: int = 42,
                   **kwargs):
    """
    向 HData 添加一个非 Hi-C 组学视图（如 RNA、METH、ATAC）。

    用法一：直接传入矩阵
        stark.pp.add_omics_view(hdata, view_key="rna", modality="rna",
                                pca=my_pca_matrix, umap=my_umap_matrix)

    用法二：从 AnnData 提取（obsm 中已有降维结果）
        stark.pp.add_omics_view(hdata, view_key="rna", modality="rna",
                                adata=rna_adata,
                                obsm_pca_key="X_pca", obsm_umap_key="X_umap")

    用法三：传入 AnnData.X，自动执行 PCA + UMAP
        stark.pp.add_omics_view(hdata, view_key="rna", modality="rna",
                                adata=rna_adata)

    参数:
        hdata:         HData 对象
        view_key:      视图唯一标识，如 "rna", "meth", "meth_50000"
        modality:      模态类型，如 MODALITY_RNA / "rna"
        pca:           (N, d) ndarray，已计算好的 PCA 结果（可选）
        umap:          (N, 2) ndarray，已计算好的 UMAP 结果（可选）
        embedding:     (N, d) ndarray，原始嵌入向量（可选）
        adata:         AnnData 对象，用于自动提取特征
        obsm_pca_key:  adata.obsm 中 PCA 的键名
        obsm_umap_key: adata.obsm 中 UMAP 的键名
        resolution:    分辨率（METH 等基因组分辨率级别视图使用）
        data_dir:      原始数据路径（仅记录元信息）
        n_components:  自动计算 PCA 时保留的方差比例（0~1）
        random_state:  随机种子
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    # 从 AnnData.obsm 中提取已有降维结果
    if adata is not None:
        if pca is None and obsm_pca_key in adata.obsm:
            pca = adata.obsm[obsm_pca_key]
        if umap is None and obsm_umap_key in adata.obsm:
            umap = adata.obsm[obsm_umap_key]
        if embedding is None and 'X_embedding' in adata.obsm:
            embedding = adata.obsm['X_embedding']

    # PCA 仍为空 → 从原始矩阵计算
    if pca is None and adata is not None and adata.X is not None:
        print(f"[add_omics_view:{view_key}] 未找到预计算 PCA，自动执行 PCA (n_components={n_components})...")
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_scaled)
        print(f"    PCA 完成，保留 {pca.shape[1]} 维。")

    # UMAP 仍为空 → 从 PCA 计算
    if umap is None and pca is not None:
        print(f"[add_omics_view:{view_key}] 未找到预计算 UMAP，自动执行 UMAP...")
        umap = UMAP(random_state=random_state, min_dist=1).fit_transform(pca)
        print(f"    UMAP 完成。")

    hdata.add_view(
        view_key=view_key,
        modality=modality,
        pca=pca,
        umap=umap,
        embedding=embedding,
        resolution=resolution,
        data_dir=data_dir,
        **kwargs,
    )

    n_cells_info = pca.shape[0] if pca is not None else "?"
    print(f"✅ 视图 [{view_key}] (modality={modality}) 已添加，细胞数={n_cells_info}。")