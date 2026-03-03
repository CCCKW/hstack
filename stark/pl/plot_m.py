import math
import cooler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.colors import LogNorm
def plot_basic_purity(hdata,figsize=(8, 6)):
    """可视化 1: 基础纯度柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='CellType_purity', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('Basic Purity by Cell Type')
    plt.tight_layout()
    plt.show()


def plot_views(hdata, label=None, ncols=3):
    """
    绘制多视图 UMAP 散点图，支持根据视图数量动态网格排版。
    
    参数:
    - hdata: HData 对象
    - label: 外部传入的标签数组或 obs 中的列名 (可选)。
    - ncols: 每行展示的图表数量，默认为 3。
    """
    # 1. 确定要使用的标签数据
    if label is not None:
        if isinstance(label, str) and label in hdata.obs.columns:
            plot_labels = hdata.obs[label].values
        else:
            plot_labels = np.asarray(label)
    else:
        if hdata.obs.empty or 'label' not in hdata.obs:
            raise ValueError("未提供 label 且 hdata.obs 中没有默认标签。")
        plot_labels = hdata.obs['label'].values

    if len(plot_labels) != hdata.n_cells:
        raise ValueError(f"提供的标签长度 ({len(plot_labels)}) 与细胞数 ({hdata.n_cells}) 不匹配。")

    # 2. 生成颜色映射
    unique_labels = sorted(list(set(plot_labels)))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colors = [color_map[lbl] for lbl in plot_labels]
    
    # 3. 动态获取所有视图
    views = list(hdata.views_umap.keys())
    num_views = len(views)
    
    if num_views == 0:
        print("没有找到任何 UMAP 数据 (hdata.views_umap 为空)。")
        return
        
    nrows = math.ceil(num_views / ncols)
    actual_ncols = min(num_views, ncols)
    
    fig, axes = plt.subplots(nrows, actual_ncols, figsize=(actual_ncols * 5, nrows * 5))
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, res in enumerate(views):
        ax = axes[i]
        ax.scatter(hdata.views_umap[res][:,0], hdata.views_umap[res][:,1], c=colors, s=5)
        ax.set_title(f'View: {res}')
        
        # 关闭坐标轴刻度让图面更干净
        ax.set_xticks([])
        ax.set_yticks([])
        
    # 隐藏多余的子图
    for j in range(num_views, len(axes)):
        axes[j].axis('off')

    # 添加 legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[lbl], markersize=8, label=lbl) 
               for lbl in unique_labels]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()



def plot_depth_distribution(hdata):
    """
    绘制深度分布直方图。
    完全还原您在 pipe.py 中的深度分布画图逻辑。
    """
    if 'depth' not in hdata.obs:
        raise ValueError("缺少深度信息，请先运行 sk.pp.process_and_load(hdata)")
        
    plt.figure(figsize=(8, 5))
    mean_depth = hdata.obs['depth'].mean()
    print(f'mean_depth:{mean_depth/1e7:.2f}M')
    
    sns.histplot(hdata.obs['depth'], bins=30, kde=True)
    plt.axvline(mean_depth, color='red', linestyle='--', label=f'Mean Depth: {mean_depth:.2f}')
    
    plt.title('Depth Distribution')
    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.show()

# ==========================================================
# 2. 模型核心结果可视化 (代理底层 plotting.py 中的方法)
# ==========================================================



def plot_ep_score(hdata,figsize=(12, 6)):
    """可视化 3: 最终评估得分 EP_v2 柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='EP_v2', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('EP v2 Score by Cell Type (Corrected & Penalized)')
    plt.tight_layout()
    plt.show()


def plot_umap_assignment(hdata,resolution=None,  figsize=(7,7)):
    """
    可视化 4: 单点 UMAP 散点图 (按模型分配的 Metacell ID 染色)
    注: 该图只依赖模型结果，可在 calculate_metrics 之前调用
    """
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    
    labels = hdata.obs['label'].values
    
    umap_coords = hdata.views_umap[resolution]
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], 
                    hue=labels, palette='tab20', s=5, legend=False, rasterized=True)
    plt.title("UMAP: Metacell Assignments")
    plt.axis('off')
    plt.tight_layout()
    plt.show()





def plot_umap_comparison(hdata, resolution=None,  figsize=(14, 6)):

        """
        可视化 5: 双点 UMAP 对比图 (左侧为预测平滑的类型，右侧为真实单细胞类型)
        """
        if resolution is None:
            raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")

        umap_coords = hdata.views_umap[resolution]
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        df = hdata.uns['eval_df_cache']
        
        # 获取所有唯一的细胞类型并排序，统一左右两图的颜色映射标准
        unified_hue_order = sorted(df['CellType'].dropna().unique())
   
        # 左图：分配的 Metacell 类型
        sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], 
                        hue=df['meta_lb'], hue_order=unified_hue_order, 
                        palette='tab20', s=5, ax=ax[0], legend=False, rasterized=True)
        ax[0].set_title("UMAP: Metacell Imputed Cell Types")
      
        
        # 右图：真实单细胞类型
        sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], 
                        hue=df['CellType'], hue_order=unified_hue_order, 
                        palette='tab20', s=5, ax=ax[1], legend=True, rasterized=True)
        ax[1].set_title("UMAP: Original Cell Types")

        
        # 图例放右侧防止遮挡
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)
        plt.tight_layout()
        plt.show()
 





def plot_metacell_sizes(hdata, figsize=(8, 6), bins=20):
    purity_df_ = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    plt.hist(purity_df_['cell_num'], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Metacell Sizes')
    plt.xlabel('Number of Cells')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
        
        



def plot_initialization(hdata, resolution=None, title="Initialization Waypoints",figsize=(8, 6)):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    umap_coords = hdata.views_umap[resolution]
    
    waypoints = hdata.model.waypoints
    
    plt.figure(figsize=figsize)
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
    wp_coords = umap_coords[waypoints, :]
    plt.scatter(wp_coords[:, 0], wp_coords[:, 1], c='red', s=60, edgecolors='black', linewidth=1, label='Initial Waypoints', zorder=10)
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.show()

def plot_specific_metacell(hdata, metacell_id,resolution=None, figsize=(8, 6)):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell'].values
    umap_coords = hdata.views_umap[resolution]
    indices = np.where(labels == metacell_id)[0]
    
    plt.figure(figsize=figsize)
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.3, label='Background')
    
    if len(indices) > 0:
        target_coords = umap_coords[indices]
        plt.scatter(target_coords[:, 0], target_coords[:, 1], c='red', s=20, label=f'Metacell {metacell_id}')
        center = np.mean(target_coords, axis=0)
        plt.scatter(center[0], center[1], c='black', marker='x', s=100, linewidth=2, label='Centroid')
        
    plt.title(f"Visual Diagnosis: Metacell {metacell_id}")
    plt.legend()
    plt.show()






def plot_metacells(hdata,resolution=None, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell']
    umap_coords = hdata.views_umap[resolution]
    metacell_coords = []
    metacell_counts = []
    present_indices = np.unique(labels)
    
    for k in present_indices:
        indices = np.where(labels == k)[0]
        metacell_coords.append(np.mean(umap_coords[indices], axis=0))
        metacell_counts.append(len(indices))
    
    metacell_coords = np.array(metacell_coords)
    metacell_counts = np.array(metacell_counts)
    
    if len(metacell_counts) == 0:
        print("警告: 没有发现活跃的 metacells。")
        return

    if len(metacell_counts) > 1 and metacell_counts.max() > metacell_counts.min():
        norm_sizes = (metacell_counts - metacell_counts.min()) / (metacell_counts.max() - metacell_counts.min())
        plot_sizes = min_size + norm_sizes * (max_size - min_size)
    else:
        plot_sizes = np.full(len(metacell_counts), (min_size + max_size) / 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
    plt.scatter(metacell_coords[:, 0], metacell_coords[:, 1], 
                c='blue', s=plot_sizes, edgecolors='white', linewidth=1, alpha=0.8, zorder=10)
    
    min_c = metacell_counts.min()
    max_c = metacell_counts.max()
    mid_c = int((min_c + max_c) / 2)
    legend_sizes = [min_size, (min_size+max_size)/2, max_size]
    legend_labels = [f'{min_c} cells', f'{mid_c} cells', f'{max_c} cells']
    
    handles = []
    for s, l in zip(legend_sizes, legend_labels):
        handles.append(plt.scatter([], [], c='blue', alpha=0.8, s=s, edgecolors='white', label=l))
    handles.append(plt.scatter([], [], c='lightgrey', s=20, label='Single Cells'))
    
    plt.legend(handles=handles, title="Metacell Size (Count)", loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.5, borderpad=1)
    
    if show_idx:
        for i, k in enumerate(present_indices):
            plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(k), 
                        fontsize=10, ha='center', va='center', color='black', fontweight='bold', zorder=20)
    
    plt.title(f"{title}\n(Metacells: {len(metacell_coords)}, Count Range: {min_c}-{max_c})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()









def plot_metacell_heatmap(hdata, metacell_id, chrom, start, end, resolution, balance=True, cmap='Reds', vmin=None, vmax=None,log1p=True,fill_diagonal_zero=True,**kwargs):
    """
    可视化单个 Metacell 在指定区间的高分辨率 Hi-C 热图。
    
    参数:
    - hdata: 核心数据对象
    - metacell_id: 目标 metacell 的 ID (字符串)
    - chrom: 染色体名称 (如 'chr1')
    - start: 起始位置
    - end: 终止位置
    - resolution: mcool 文件中的分辨率
    - balance: 是否使用 balance 后的矩阵 (布尔值)
    """
    if 'mcool' not in hdata.metacell_data or metacell_id not in hdata.metacell_data['mcool']:
        raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mcool 记录。请确保已经运行过 aggregate_metacell_pairs。")
        
    mcool_path = hdata.metacell_data['mcool'][metacell_id]
    uri = f"{mcool_path}::/resolutions/{resolution}"
    
    # 读取数据
    clr = cooler.Cooler(uri)
    mat = clr.matrix(balance=balance).fetch((chrom, start, end))
    mat = np.nan_to_num(mat) # 将 balance 可能产生的 NaN 替换为 0
    if log1p:
        mat = np.log1p(mat)
    if fill_diagonal_zero:
        np.fill_diagonal(mat, 0)
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal') # 强制正方形
    
    if vmax is None:
        if vmin is None:
            sns.heatmap(mat,cmap='Reds',cbar=True,ax=ax)
        else:
            sns.heatmap(mat,cmap='Reds',cbar=True,ax=ax,vmin=vmin)
    else:
        if vmin is None:
            sns.heatmap(mat,cmap='Reds',cbar=True,ax=ax,vmax=vmax)
        else:
            sns.heatmap(mat,cmap='Reds',cbar=True,ax=ax,vmin=vmin,vmax=vmax)
    
    # 添加装饰
    ax.set_title(f"Metacell: {metacell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")

    
    plt.tight_layout()
    plt.show()
    return mat

def plot_celltype_heatmaps(hdata, cell_type, chrom, start, end, resolution, 
                            balance=True, ncols=4, 
                            cell_type_col='cell_type', cmap='Reds',
                            log1p=True, fill_diagonal_zero=True,
                             vmax=None, vmin=None):
    """
    可视化指定细胞类型下所有 Metacell 的 Hi-C 热图 (以网格形式展示)。
    
    参数:
    - hdata: 核心数据对象
    - cell_type: 目标细胞类型名称
    - chrom: 染色体名称
    - start: 起始位置
    - end: 终止位置
    - resolution: mcool 文件中的分辨率
    - balance: 是否使用 balance 后的矩阵
    - ncols: 网格每一行展示的图表数量
    - cell_type_col: hdata.metacell_obs 中记录细胞类型的列名，默认为 'cell_type'
    """
    # 1. 获取对应细胞类型的所有 Metacell IDs
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacell_obs 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
        
    target_obs = hdata.metacells[hdata.metacells[cell_type_col] == cell_type]
    m_ids = target_obs.index.tolist()
    
    if not m_ids:
        print(f"未找到细胞类型为 '{cell_type}' 的 Metacell，请检查名称是否正确。")
        return
        
    print(f"共找到 {len(m_ids)} 个属于 '{cell_type}' 的 Metacells, 准备渲染...")
    
    # 2. 初始化网格画布
    nrows = math.ceil(len(m_ids) / ncols)
    # 根据行列数动态调整总画布大小，确保每个子图为正方形的视觉基础
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    
    if isinstance(axes, plt.Axes):
        axes = [axes] # 单个子图情况
    else:
        axes = axes.flatten() # 展平方便遍历
        
    # 3. 遍历渲染子图
    for i, m_id in enumerate(m_ids):
        ax = axes[i]
        ax.set_aspect('equal') # 强制子图内部坐标系严格正方形
        
        if m_id not in hdata.metacell_data.get('mcool', {}):
            ax.set_title(f"{m_id}\n(No mcool data)")
            ax.axis('off')
            continue
            
        try:
            mcool_path = hdata.metacell_data['mcool'][m_id]
            uri = f"{mcool_path}::/resolutions/{resolution}"
            
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
            mat = np.nan_to_num(mat)
            if log1p:
                mat = np.log1p(mat)
            if fill_diagonal_zero:
                np.fill_diagonal(mat, 0)

            if vmax is None:
                if vmin is None:
                    sns.heatmap(mat,cmap=cmap,cbar=True,ax=ax)
                else:
                    sns.heatmap(mat,cmap=cmap,cbar=True,ax=ax,vmin=vmin)
            else:
                if vmin is None:
                    sns.heatmap(mat,cmap=cmap,cbar=True,ax=ax,vmax=vmax)
                else:
                    sns.heatmap(mat,cmap=cmap,cbar=True,ax=ax,vmin=vmin,vmax=vmax)
            
            
            ax.set_title(m_id)
            
            # 关闭多图展示时的冗余坐标刻度，保持画面清爽
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.set_title(f"{m_id}\n(Error)")
            ax.axis('off')
            
    # 4. 隐藏多余的空白子图占位
    for j in range(len(m_ids), len(axes)):
        axes[j].axis('off')
        
    # 设置主标题
    plt.suptitle(f"Cell Type: {cell_type} | Region: {chrom}:{start}-{end} @ {resolution//1000}kb", 
                 y=1.02, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()