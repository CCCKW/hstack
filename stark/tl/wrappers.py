from ..utils.rec_num import recommend_by_leiden
from ..utils.model import MultiViewSEACells  
import numpy as np



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
    hdata.model.plot_initialization(hdata.views_umap[ref_view_res], title="Check K-Means++ Initialization")

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


def evaluate(hdata):
    """
    步骤 9: 评估模型并计算纯度
    """
    if hdata.model is None or 'metacell' not in hdata.obs:
        raise ValueError("模型尚未拟合，请先运行 sk.tl.fit(hdata)")
        
    true_labels = hdata.obs['label'].values
    purity_df = hdata.model.calculate_metrics(true_labels)
    metrics_summary = hdata.model.get_metrics_summary()
    
    hdata.uns['purity_df'] = purity_df
    hdata.uns['metrics'] = metrics_summary
    
    # ==============================================================
    # 新增：将 purity 核心指标无缝追加到现有的 hdata.metacells 中
    # ==============================================================
    # 过滤掉与基础属性重复的列 (cell_num 对应 cell_count, CellType 对应 dominant_label)
    cols_to_add = [c for c in purity_df.columns if c not in ['CellType', 'cell_num']]
    
    if hdata.metacells.empty:
        hdata.metacells = purity_df.copy() # 防御性编程：如果用户没正常走 fit
    else:
        for col in cols_to_add:
            hdata.metacells[col] = purity_df[col]
            
    print("✅ 评估指标计算完成，纯度得分(EP_v2等)已同步至 hdata.metacells。")
    return purity_df, metrics_summary



