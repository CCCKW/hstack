from ..utils.rec_num import recommend_by_leiden
from ..utils.model import MultiViewSEACells  
import numpy as np
import pandas as pd
def calculate_overmerging_metrics(cell_to_metacell, true_labels):
    """
    评估 Metacell 划分中是否出现“巨大且不纯”的过度融合现象 (Over-merging / Hub effect)。
    
    参数:
    cell_to_metacell (array-like): 长度为 N 的数组，记录每个单细胞被分配到的 metacell ID。
    true_labels (array-like): 长度为 N 的数组，记录每个单细胞的真实细胞类型 (Ground truth)。
    
    返回:
    dict: 包含 WCOS 和 HWIS 两个 [0, 1] 之间的指标。值越接近 1 越好，越接近 0 问题越严重。
    """
    df = pd.DataFrame({
        'metacell_id': cell_to_metacell,
        'true_label': true_labels
    })
    
    total_cells = len(df)
    
    # 用于记录每个 metacell 的惩罚值
    penalties = []
    squared_penalties = []
    
    # 遍历每一个 metacell
    for mc_id, group in df.groupby('metacell_id'):
        size = len(group)
        size_fraction = size / total_cells  # 该 metacell 占总数据的比例 (0 到 1)
        
        # 计算该 metacell 的纯度 (Purity)
        # 纯度 = 该 metacell 中数量最多的真实细胞类型的占比
        mode_count = group['true_label'].value_counts().iloc[0]
        purity = mode_count / size
        impurity = 1.0 - purity  # 不纯度 (0 到 1)
        
        # 计算惩罚项
        penalty = size_fraction * impurity
        squared_penalty = (size_fraction ** 2) * impurity
        
        penalties.append(penalty)
        squared_penalties.append(squared_penalty)
        
    # ---------------------------------------------------------
    # 指标 1: WCOS (Worst-Case Overmerging Score)
    # 专注于最糟糕的那单个 metacell。寻找 (尺寸最大且最不纯) 的极端值。
    # 如果有一个 300/700 的 metacell，纯度只有 0.2，那么 penalty = (300/700) * 0.8 ≈ 0.34
    # WCOS = 1 - 0.34 = 0.66 (显著下降)
    # ---------------------------------------------------------
    max_penalty = np.max(penalties) if penalties else 0
    wcos = 1.0 - max_penalty
    
    # ---------------------------------------------------------
    # 指标 2: HWIS (Hub-Weighted Impurity Score)
    # 评估全局的健康度。利用平方操作严厉惩罚尺寸分布的不均（大尺寸的权重被指数级放大）。
    # 这个指标能很好地反映出算法是否在依赖几个“垃圾桶” metacell 来吸收难以分类的细胞。
    # ---------------------------------------------------------
    agg_squared_penalty = np.sum(squared_penalties) if squared_penalties else 0
    hwis = 1.0 - agg_squared_penalty
    
    return {
        "WCOS": float(wcos),
        "HWIS": float(hwis)
    }


def recommend_metacell_num(hdata, target_depth_min=20*1e6, target_depth_max=40*1e6, resolution_param=2.0, n_neighbors=15, ref_view=1000000):
    """
    步骤 2: 推荐 MetaCell 范围并保存到 hdata
    完全调用您底层的 recommend_by_leiden，不改变任何算法逻辑。
    """
    # 从 HData 的统一表格中抽取 depth
    depth_array = hdata.obs['depth'].values 
    
    min_k, max_k = recommend_by_leiden(
        depth_array=depth_array,
        features=hdata.views_pca[ref_view], 
        target_depth_min=target_depth_min,  
        target_depth_max=target_depth_max,  
        resolution=resolution_param,
        n_neighbors=n_neighbors
    )
    
    # 将推荐的范围保存到非结构化字典中备用
    hdata.uns['recommended_k'] = (min_k, max_k)
    return min_k, max_k

def init_model(hdata, n_metacells, **kwargs):
    """
    步骤 3: 初始化模型参数，实例化 MultiViewSEACells 并将其挂载到 hdata
    """
    # 保留您在 pipe.py 中的默认参数
    default_params = {
        'lambda_sparse': 0.0000,
        'lambda_balance': 0.01,
        'lambda_consistency': 0.001,
        'max_iter': 100,
        'min_size_threshold': 0.002,
        'respawn_interval': 10,
        'split_metric': 'pca'
    }
    default_params.update(kwargs)
    
    # 实例化您原始的类，不修改任何内部结构
    hdata.model = MultiViewSEACells(n_metacells=n_metacells, **default_params)
    print(f"✅ 模型参数初始化完成，目标 MetaCell 数量: {n_metacells}")

def compute_kernels(hdata):
    """
    步骤 4: 计算核矩阵
    严格按照您 pipe.py 中的顺序输入: 100K, 500K, 1M (对应 view1, view2, view3)
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化，请先运行 sk.tl.init_model(hdata, ...)")
        
    pca_list = [
        hdata.views_pca[100000],   # view1_pca
        hdata.views_pca[500000],   # view2_pca
        hdata.views_pca[1000000]   # view3_pca
    ]
    
    # 完全调用底层的核矩阵计算
    hdata.model.compute_kernels(pca_list, save_dir=None)

def initialize_waypoints(hdata, seed=32, n_micro_clusters=None, ref_view_res=500000):
    """
    步骤 5: 模型 initialize + 顺带调用其可视化确认 waypoint
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化")
        
    if n_micro_clusters is None:
        n_micro_clusters = hdata.model.n_metacells
        
    # 调用底层 initialize
    hdata.model.initialize(seed=seed, data_type='kernel', n_micro_clusters=n_micro_clusters)
    
    # 按照您的流程，这一步直接出图确认

def fit(hdata, n_threads=10):
    """
    步骤 6: 进行模型拟合，并将结果（metacell 分配标签）持久化存回 HData.obs
    同时初始化 Metacell 的基础统计属性。
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化")
        
    hdata.model.fit(n_threads=n_threads)
    hdata.obs['metacell'] = hdata.model.labels
    
    # ==============================================================
    # 新增：在获取 metacell 标签后，立刻初始化 hdata.metacells 基础表
    # ==============================================================
    meta_stats = hdata.obs.groupby('metacell').agg({
        'depth': ['sum', 'count', 'mean']
    })
    meta_stats.columns = ['total_depth', 'cell_count', 'mean_depth']
    
    if 'label' in hdata.obs.columns:
        def get_dominant(x): return x.value_counts().index[0]
        meta_stats['dominant_label'] = hdata.obs.groupby('metacell')['label'].apply(get_dominant)
        
    hdata.metacells = meta_stats
    print("✅ 模型拟合完成，Metacell 标签已保存，基础属性(深度、组成等)已初始化至 hdata.metacells。")
    
    
def calculate_metrics(hdata, cell_types):
        """
        步骤1: 计算核心评估指标，并将其缓存为模型属性 (纯计算，不绘图)
        
        参数:
        - cell_types: array-like, 细胞的真实类型标签
        
        返回:
        - purity_df: 包含各项纯度和大小评估指标的 DataFrame
        """

        print("\n" + "=" * 60)
        print("正在计算评估指标...")
        print("=" * 60)
        labels = hdata.obs['metacell'].values
        # 整理基础 DataFrame
        df = pd.DataFrame({'CellType': cell_types, 'Metacell': labels})
        
        overmerge = calculate_overmerging_metrics(df['Metacell'], df['CellType'])
        wcos = overmerge['WCOS']
        hwis = overmerge['HWIS']
        
        
        def celltype_frac(x):
            val_counts = x['CellType'].value_counts()
            return val_counts.values[0] / val_counts.values.sum()
            
        def dominant_celltype(x):
            return x['CellType'].value_counts().index[0]

        # 聚合计算
        celltype_fraction = df.groupby("Metacell").apply(celltype_frac)
        celltype_dom = df.groupby("Metacell").apply(dominant_celltype)
        cell_num = df.groupby("Metacell").count()['CellType']

        purity = pd.concat([celltype_dom, celltype_fraction, cell_num], axis=1)
        purity.columns = ['CellType', 'CellType_purity', 'cell_num']

        # 动态计算惩罚因子与基线调整
        avg_size = purity['cell_num'].mean()
        thre = 2 * avg_size
        
        # 1. 过小惩罚
        purity['w_min'] = 1 - (1 / np.sqrt(purity['cell_num']))
        purity.loc[purity['cell_num'] == 1, 'w_min'] = 0.0 # 处理特例
        
        # 2. 过大惩罚
        excess_ratio = (purity['cell_num'] - thre) / avg_size
        excess_ratio = excess_ratio.clip(lower=0) 
        purity['w_max'] = 1 / (1 + excess_ratio)
        
        # 3. 机会校正基线
        num_unique_types = df['CellType'].nunique()
        baseline = 1.0 / num_unique_types
        purity['P_adj'] = (purity['CellType_purity'] - baseline) / (1 - baseline)
        purity['P_adj'] = purity['P_adj'].clip(lower=0)
        
        # 4. 最终核心 EP_v2
        purity['EP_v2'] = purity['P_adj'] * purity['w_min'] * purity['w_max']
        
        # 记录内部属性
        mean_purity_ = purity['CellType_purity'].mean()
        global_score_ = (purity['EP_v2'] * purity['cell_num']).sum() / purity['cell_num'].sum()
        
        # 计算 Accuracy 并映射标签以便画图
        hash_meta = purity['CellType'].to_dict()
        df['meta_lb'] = df['Metacell'].map(hash_meta)
        accuracy_ = (df['CellType'] == df['meta_lb']).sum() / df.shape[0]

  
        print(f"✅ 指标计算完成！(发现 {num_unique_types} 种细胞类型)")
        return purity, df, avg_size, thre, wcos, hwis


def evaluate(hdata, true_labels ):
    """
    步骤 9: 评估模型并计算纯度
    """

        
    purity_df, eval_df_cache, avg_size_cache, thre_cache , wcos, hwis= calculate_metrics(hdata, true_labels)
    accuracy = (eval_df_cache['CellType'] == eval_df_cache['meta_lb']).sum() / eval_df_cache.shape[0]
    global_score = (purity_df['EP_v2'] * purity_df['cell_num']).sum() / purity_df['cell_num'].sum()
    print("-" * 40)
    print(f"简单平均纯度 (Mean Purity)  : {purity_df['CellType_purity'].mean():.4f}")
    print(f"模型准确率 (Accuracy)      : {accuracy:.4f}")
    print(f"全局加权分 (Global Score)  : {global_score:.4f}")
    print(f"过度融合指标 (WCOS)       : {wcos:.4f}")
    print(f"Hub 权重不纯度 (HWIS)     : {hwis:.4f}")
    print("-" * 40)
        
    metrics_summary = {
            'mean_purity': purity_df['CellType_purity'].mean(),
            'accuracy': accuracy,
            'global_score': global_score,
            'WCOS': wcos,
            'HWIS': hwis
        }


    
    hdata.uns['purity_df'] = purity_df
    hdata.uns['metrics'] = metrics_summary
    hdata.uns['eval_df_cache'] = eval_df_cache
    hdata.uns['avg_size_cache'] = avg_size_cache
    hdata.uns['thre_cache'] = thre_cache
    hdata.uns['accuracy'] = accuracy
    hdata.uns['global_score'] = global_score
    hdata.uns['wcos'] = wcos
    hdata.uns['hwis'] = hwis
    
    
    # ==============================================================
    # 新增：将 purity 核心指标无缝追加到现有的 hdata.metacells 中
    # ==============================================================
    # 过滤掉与基础属性重复的列 (cell_num 对应 cell_count, CellType 对应 dominant_label)
    cols_to_add = [c for c in purity_df.columns if c not in ['CellType', 'cell_num']]
    
    if 'hdata.metacells' not in hdata.__dict__ or hdata.metacells.empty:
        hdata.metacells = purity_df.copy()
    
    if hdata.metacells.empty:
        hdata.metacells = purity_df.copy() # 防御性编程：如果用户没正常走 fit
    else:
        for col in cols_to_add:
            hdata.metacells[col] = purity_df[col]
            
    print("✅ 评估指标计算完成，纯度得分(EP_v2等)已同步至 hdata.metacells。")
    return purity_df, metrics_summary



