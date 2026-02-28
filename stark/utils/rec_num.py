import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from typing import Tuple

def recommend_n_metacells_global(
    depth_array: np.ndarray, 
    target_depth_min: float = 500e6, 
    target_depth_max: float = 800e6,
    min_cells_per_metacell: int = 10,
    plot_simulation: bool = True
) -> Tuple[int, int]:
    """
    基于全局深度推荐 metacell 数量范围 (直接接收 numpy 数组)。
    """
    cell_depths = np.asarray(depth_array).flatten()
    total_depth = np.sum(cell_depths)
    n_cells = len(cell_depths)
    
    # 根据最大目标深度计算最小 K
    min_k = int(np.round(total_depth / target_depth_max))
    # 根据最小目标深度计算最大 K
    max_k = int(np.round(total_depth / target_depth_min))
    
    # 边界约束检查
    upper_bound_k = n_cells // min_cells_per_metacell
    min_k = max(1, min(min_k, upper_bound_k))
    max_k = max(1, min(max_k, upper_bound_k))
    if min_k > max_k:
        min_k = max_k
        
    print(f"🌍 全局推荐结果: n_metacells 范围 = [{min_k}, {max_k}]")
    print(f"   (对应预期平均深度从 {total_depth/max_k:.2e} 到 {total_depth/min_k:.2e})")
    
    if plot_simulation:
        # 使用中间值 K 进行模拟展示
        mid_k = (min_k + max_k) // 2
        print(f"   使用中间值 K={mid_k} 进行模拟分布展示...")
        _simulate_and_plot(cell_depths, mid_k, target_depth_min, target_depth_max)
        
    return min_k, max_k

def recommend_by_leiden(
    depth_array: np.ndarray,
    features: np.ndarray,
    target_depth_min: float = 500e6,  # 目标深度下限
    target_depth_max: float = 800e6,  # 目标深度上限
    min_metacells_per_cluster: int = 1,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    plot_result: bool = True
) -> Tuple[int, int]:
    """
    使用 Leiden 聚类并基于每个类的深度来推荐 metacell 数量区间。
    """
    cell_depths = np.asarray(depth_array).flatten()
    
    if len(cell_depths) != features.shape[0]:
        raise ValueError(f"深度数组的细胞数 ({len(cell_depths)}) 与特征矩阵的细胞数 ({features.shape[0]}) 不匹配！")
        
    print(f"正在构建 AnnData 并计算近邻图 (n_neighbors={n_neighbors})...")
    adata = ad.AnnData(X=features,  dtype=features.dtype)
    
    # 2. 修复 macOS 环境下由 OpenMP 导致的内核崩溃问题
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
    print(f"正在执行 Leiden 聚类 (resolution={resolution})...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    cluster_labels = adata.obs['leiden'].values
    clusters = np.unique(cluster_labels)
    n_clusters = len(clusters)
    print(f"✅ 聚类完成，共发现 {n_clusters} 个聚类簇。")

    # === 计算推荐 K 值区间 ===
    cluster_stats = []
    total_min_k = 0
    total_max_k = 0
    
    for c in clusters:
        mask = cluster_labels == c
        c_cells = mask.sum()
        c_depth = np.sum(cell_depths[mask])
        
        # 按照该簇的总深度切分出 min_K 和 max_K
        min_k_raw = int(np.round(c_depth / target_depth_max))
        max_k_raw = int(np.round(c_depth / target_depth_min))
        
        # 保护小类，确保每个类至少有指定的 metacell 数量
        c_min_k = max(min_metacells_per_cluster, min_k_raw)
        c_max_k = max(min_metacells_per_cluster, max_k_raw)
        if c_min_k > c_max_k:
            c_min_k = c_max_k
            
        total_min_k += c_min_k
        total_max_k += c_max_k
        
        cluster_stats.append({
            'Cluster': c,
            'Cells': c_cells,
            'Total_Depth': c_depth,
            'Min_K': c_min_k,
            'Max_K': c_max_k
        })
        
    stats_df = pd.DataFrame(cluster_stats).sort_values('Cells', ascending=False)
    
    print(f"\n💡 基于 Leiden 聚类的推荐结果: 最终总 n_metacells 范围 = [{total_min_k}, {total_max_k}]")
    
    # === 可视化 ===
    if plot_result:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=stats_df, x='Cluster', y='Cells', color='lightgrey', ax=ax1, label='Cell Count')
        ax1.set_ylabel('Number of Cells', color='grey', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='grey')
        plt.xticks(rotation=45)
        
        ax2 = ax1.twinx()
        
        # 绘制 K 值区间阴影
        x_positions = np.arange(len(stats_df))
        ax2.fill_between(x_positions, stats_df['Min_K'], stats_df['Max_K'], color='red', alpha=0.2, label='K Range')
        
        # 绘制中间基准线
        mid_k = (stats_df['Min_K'] + stats_df['Max_K']) / 2
        sns.lineplot(x=x_positions, y=mid_k, color='red', marker='o', ax=ax2, label='Mid K', linewidth=2, markersize=8)
        
        ax2.set_ylabel('Allocated Metacells (K)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'Metacell Allocation per Leiden Cluster (Total K = {total_min_k} ~ {total_max_k})', fontsize=14)
        fig.tight_layout()
        plt.show()

    return total_min_k, total_max_k

def _simulate_and_plot(cell_depths, k, min_target, max_target):
    """
    (内部函数) 模拟均等划分细胞下的 Metacell 深度分布
    """
    np.random.seed(42)
    shuffled_depths = np.random.permutation(cell_depths)
    simulated_metacells = np.array_split(shuffled_depths, k)
    simulated_depths = [np.sum(mc) for mc in simulated_metacells]
    
    p05 = np.percentile(simulated_depths, 5)
    p95 = np.percentile(simulated_depths, 95)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_depths, kde=True, bins=30, color='#4A90E2', edgecolor='white')
    plt.axvspan(min_target, max_target, color='green', alpha=0.15, label='Target Depth Range')
    plt.axvline(min_target, color='green', linestyle='--', linewidth=2)
    plt.axvline(max_target, color='green', linestyle='--', linewidth=2)
    plt.axvline(p05, color='red', linestyle=':', label=f'5th Percentile ({p05:.2e})')
    plt.axvline(p95, color='red', linestyle=':', label=f'95th Percentile ({p95:.2e})')
    
    plt.title(f'Simulated Metacell Depth Distribution (using mid_k={k})', fontsize=14)
    plt.xlabel('Metacell Total Depth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()