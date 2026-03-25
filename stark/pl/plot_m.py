import math
import cooler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.colors import LogNorm
def plot_basic_purity(hdata,figsize=(8, 6), save_path=None, dpi=300):
    """可视化 1: 基础纯度柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='CellType_purity', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('Basic Purity by Cell Type')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_views(hdata, label=None, ncols=3, save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()



def plot_depth_distribution(hdata, save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# ==========================================================
# 2. 模型核心结果可视化 (代理底层 plotting.py 中的方法)
# ==========================================================



def plot_ep_score(hdata,figsize=(12, 6), save_path=None, dpi=300):
    """可视化 3: 最终评估得分 EP_v2 柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='EP_v2', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('EP v2 Score by Cell Type (Corrected & Penalized)')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_umap_assignment(hdata,resolution=None,  figsize=(7,7), save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()





def plot_umap_comparison(hdata, resolution=None,  figsize=(14, 6), save_path=None, dpi=300):

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
        if 'save_path' in locals() and save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
 





def plot_metacell_sizes(hdata, figsize=(8, 6), bins=20, save_path=None, dpi=300):
    purity_df_ = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    plt.hist(purity_df_['cell_num'], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Metacell Sizes')
    plt.xlabel('Number of Cells')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
        
        



def plot_initialization(hdata, resolution=None, title="Initialization Waypoints",figsize=(8, 6), save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_specific_metacell(hdata, metacell_id,resolution=None, figsize=(8, 6), save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()






def plot_metacells(hdata,resolution=None, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False, save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_metacells2(hdata, resolution=None, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False, label_col='label', cell_alpha=0.3, metacell_alpha=0.8, save_path=None, dpi=300):
    """
    可视化 Metacell 在 UMAP 上的位置，并用主导细胞类型 (dominant label) 染色，
    同时将底层的单细胞也一并着色展示。
    """
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell']
    umap_coords = hdata.views_umap[resolution]
    metacell_coords = []
    metacell_counts = []
    metacell_dominant_labels = []
    present_indices = np.unique(labels)
    
    import pandas as pd
    has_labels = False
    if label_col in hdata.obs.columns:
        cell_labels = hdata.obs[label_col].values
        has_labels = True
        
    for k in present_indices:
        indices = np.where(labels == k)[0]
        metacell_coords.append(np.mean(umap_coords[indices], axis=0))
        metacell_counts.append(len(indices))
        if has_labels:
            dom_label = pd.Series(cell_labels[indices]).value_counts().index[0]
            metacell_dominant_labels.append(dom_label)
        else:
            metacell_dominant_labels.append("Unknown")
            
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
    
    # 提前处理颜色映射
    mc_colors = ['blue'] * len(metacell_coords)
    cell_colors_list = 'lightgrey'
    handles = []
    
    if has_labels:
        if 'eval_df_cache' in hdata.uns:
            unified_hue_order = sorted(hdata.uns['eval_df_cache']['CellType'].dropna().unique())
        else:
            unified_hue_order = sorted(pd.Series(cell_labels).dropna().unique())
            
        palette = sns.color_palette('tab20', n_colors=len(unified_hue_order))
        color_map = {lbl: palette[i] for i, lbl in enumerate(unified_hue_order)}
        
        mc_colors = [color_map.get(lbl, 'blue') for lbl in metacell_dominant_labels]
        cell_colors_list = [color_map.get(lbl, 'lightgrey') for lbl in cell_labels]
        
        # 添加类别图例
        for lbl in unified_hue_order:
            handles.append(plt.scatter([], [], c=[color_map[lbl]], s=100, edgecolors='none', linewidth=0, label=lbl))
        # 空白分隔
        handles.append(plt.scatter([], [], c='white', s=0, label=''))

    # 绘制底层的单细胞
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=cell_colors_list, s=5, alpha=cell_alpha, rasterized=True, edgecolors='none', linewidth=0)

    # 绘制顶层的 Metacell
    plt.scatter(metacell_coords[:, 0], metacell_coords[:, 1],
            c=mc_colors, s=plot_sizes, edgecolors='none', linewidths=0, alpha=metacell_alpha, zorder=10)
    min_c = metacell_counts.min()
    max_c = metacell_counts.max()
    mid_c = int((min_c + max_c) / 2)
    legend_sizes = [min_size, (min_size+max_size)/2, max_size]
    legend_labels = [f'{min_c} cells', f'{mid_c} cells', f'{max_c} cells']
    
    # 尺寸图例
    for s, l in zip(legend_sizes, legend_labels):
        handles.append(plt.scatter([], [], c='gray', alpha=0.8, s=s, edgecolors='none', linewidth=0, label=l))
    handles.append(plt.scatter([], [], c='lightgrey', s=20, edgecolors='none', linewidth=0, label='Single Cells'))
    
    plt.legend(handles=handles, title="Metacell Legend", loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.0, borderpad=1)
    
    if show_idx:
        for i, k in enumerate(present_indices):
            plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(k), 
                        fontsize=10, ha='center', va='center', color='black', fontweight='bold', zorder=20)
    
    plt.title(f"{title}\n(Metacells: {len(metacell_coords)}, Count Range: {min_c}-{max_c})")
    plt.axis('off')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_metacell_heatmap(hdata, metacell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='Reds', vmin=None, vmax=None,log1p=True,fill_diagonal_zero=True,save_path=None, dpi=300, **kwargs):
    """
    可视化单个 Metacell 在指定区间的高分辨率 Hi-C 热图。
    
    参数:
    - hdata: 核心数据对象
    - metacell_id: 目标 metacell 的 ID (字符串或数字，会被强转对齐)
    - chrom: 染色体名称 (如 'chr1')
    - start: 起始位置 (bp)
    - end: 终止位置 (bp)
    - resolution: 分辨率
    - base_on: 'pair' (通过 mcool 读取) 或 'mat' (直接读取 views_mat 聚合结果)
    - balance: 是否使用 balance 后的矩阵 (布尔值，仅对 'pair' 模式生效)
    """
    if base_on == 'pair':
        if 'mcool' not in hdata.metacell_data or metacell_id not in hdata.metacell_data['mcool']:
            raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mcool 记录。请确保已经运行过 aggregate_metacell_pairs。")
            
        mcool_path = hdata.metacell_data['mcool'][metacell_id]
        try:
            uri = f"{mcool_path}::/resolutions/{resolution}"
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在 mcool 文件中读取分辨率 {resolution}。确保 pair 流程初始化了该分辨率。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus']:
        str_res = str(resolution)
        dict_key = base_on
        if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
            raise ValueError(f"未找到基于 {base_on} 聚合的分辨率 {resolution} 数据。请先运行对应的聚合函数。")
            
        mcool_dict = hdata.metacell_data[dict_key][str_res]
        if metacell_id not in mcool_dict:
             raise ValueError(f"未找到 Metacell '{metacell_id}' 的 {base_on} 聚合记录。")
             
        if chrom not in mcool_dict[metacell_id]:
             raise ValueError(f"未找到染色体 {chrom} 的聚合矩阵。")
             
        whole_chrom_mat = mcool_dict[metacell_id][chrom]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) # 将 NaN 替换为 0
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return mat

def plot_cell_of_metacell_heatmap(hdata, metacell_id, cell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='Reds', vmin=None, vmax=None, log1p=True, fill_diagonal_zero=True, save_path=None, dpi=300, **kwargs):
    """
    可视化指定 Metacell 内单个原始细胞的高分辨率 Hi-C 热图。
    
    参数:
    - hdata: 核心数据对象
    - metacell_id: 目标 metacell 的 ID (如果在 hdata.obs 中有记录可供校验，否则仅用于标题展示)
    - cell_id: 目标单细胞的 ID (在 hdata.obs.index 中的索引或名称)
    - 其他参数与 plot_metacell_heatmap 保持一致
    """
    mat = None
    if base_on == 'pair':
        # 尝试查找已生成的单细胞 cool/mcool 文件
        # 如果 cell_id 是整数索引，我们根据 hdata.data_dir 中的文件顺序推推断其对应的文件名
        cell_name = str(cell_id)
        if isinstance(cell_id, int) or str(cell_id).isdigit():
            idx = int(cell_id)
            all_files = []
            import os
            for val in sorted(os.listdir(hdata.data_dir)):
                if val.endswith('.pairs') or val.endswith('.pairs.gz'):
                    all_files.append(val)
            if idx < len(all_files):
                cell_name = all_files[idx].split('.pairs')[0]
        
        # 在相关目录中寻找匹配的 mcool/cool 文件
        mcool_path = None
        import os
        search_dirs = [hdata.data_dir, hdata.output_dir]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if f.endswith('.mcool') or f.endswith('.cool'):
                        if cell_name in f or str(cell_id) in f:
                            mcool_path = os.path.join(root, f)
                            break
                if mcool_path: break
            if mcool_path: break
            
        if not mcool_path:
            raise ValueError(f"未找到单细胞 '{cell_id}' 对应的 .mcool 或 .cool 文件，请确认已生成该文件或使用 base_on='mat'。")
            
        try:
            import cooler
            if mcool_path.endswith('.mcool'):
                uri = f"{mcool_path}::/resolutions/{resolution}"
            else:
                uri = f"{mcool_path}::/resolutions/{resolution}"
                if not cooler.fileops.is_multires_file(mcool_path):
                    uri = mcool_path
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在文件 {mcool_path} 中读取数据。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus']:
        if not hdata.views_mat:
            raise ValueError("hdata.views_mat 为空，请确认已运行预处理。")
        if resolution not in hdata.views_mat:
            raise ValueError(f"未找到分辨率 {resolution} 的 views_mat 数据。")
        if chrom not in hdata.views_mat[resolution]:
            raise ValueError(f"未找到染色体 {chrom} 的 views_mat 数据。")
            
        try:
            cell_idx = hdata.obs.index.get_loc(cell_id)
        except KeyError:
            if isinstance(cell_id, int) and cell_id < len(hdata.obs):
                cell_idx = cell_id
            else:
                raise ValueError(f"Cell ID '{cell_id}' 不在 hdata.obs.index 中。")
                
        whole_chrom_mat = hdata.views_mat[resolution][chrom][cell_idx]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        # hdata.views_mat 中通常是稀疏矩阵，需转为密集矩阵并切片
        import scipy.sparse as sp
        if sp.issparse(whole_chrom_mat):
            mat = whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
        else:
            mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat)
    if log1p:
        mat = np.log1p(mat)
    if fill_diagonal_zero:
        np.fill_diagonal(mat, 0)
        
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    if vmax is None:
        if vmin is None:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax)
        else:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin)
    else:
        if vmin is None:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmax=vmax)
        else:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax)
    
    ax.set_title(f"Metacell: {metacell_id} | Cell: {cell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return mat


def plot_celltype_heatmaps(hdata, cell_type, chrom, start, end, resolution, 
                            balance=True, base_on='pair', ncols=4, 
                            cell_type_col='cell_type', cmap='Reds',
                            log1p=True, fill_diagonal_zero=True,
                             vmax=None, vmin=None, save_path=None, dpi=300):
    """
    可视化指定细胞类型下所有 Metacell 的 Hi-C 热图 (以网格形式展示)。
    
    参数:
    - hdata: 核心数据对象
    - cell_type: 目标细胞类型名称
    - chrom: 染色体名称
    - start: 起始位置
    - end: 终止位置
    - resolution: 分辨率
    - base_on: 'pair' (通过 mcool 读取) 或 'mat' (直接读取 views_mat 聚合结果)
    - balance: 是否使用 balance 后的矩阵
    - ncols: 网格每一行展示的图表数量
    - cell_type_col: hdata.metacell_obs 中记录细胞类型的列名，默认为 'cell_type'
    """
    # 1. 获取对应细胞类型的所有 Metacell IDs
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
        
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
        
        try:
            if base_on == 'pair':
                if m_id not in hdata.metacell_data.get('mcool', {}):
                    ax.set_title(f"{m_id}\n(No mcool data)")
                    ax.axis('off')
                    continue
                mcool_path = hdata.metacell_data['mcool'][m_id]
                uri = f"{mcool_path}::/resolutions/{resolution}"
                clr = cooler.Cooler(uri)
                mat = clr.matrix(balance=balance).fetch((chrom, start, end))
                
            elif base_on in ['mat', 'mat_redist', 'mat_consensus']:
                str_res = str(resolution)
                dict_key = base_on
                if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
                    raise ValueError(f"未找到基于 {base_on} 聚合的分辨率数据。")
                mcool_dict = hdata.metacell_data[dict_key][str_res]
                if m_id not in mcool_dict or chrom not in mcool_dict[m_id]:
                    ax.set_title(f"{m_id}\n(No mat data)")
                    ax.axis('off')
                    continue
                
                whole_chrom_mat = mcool_dict[m_id][chrom]
                start_bin = int(start // resolution)
                end_bin = int(np.ceil(end / resolution))
                max_bins = whole_chrom_mat.shape[0]
                start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)
                mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
            else:
                raise ValueError("base_on 必须是 'pair' 或 'mat'")

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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# ==========================================================
# 3. 结构增强版可视化 (First-Principle O/E 校正)
# ==========================================================

from scipy.ndimage import gaussian_filter1d

def _calculate_oe(mat, log2_transform=True, pseudocount=1.0, mask_threshold=0.05):
    """
    计算 Observed/Expected (O/E) 矩阵核心算法。
    已增设[高斯平滑距离衰减曲线]与[低覆盖坏点遮蔽]功能，彻底消除由于稀疏数据带来的平行对角线锯齿条纹与贯穿式十字蓝带。
    """
    n = mat.shape[0]
    
    # 1. 坏点检测 (Coverage极低的测序死角 unmappable bins，必须拉黑，否则算 O/E 会被放大为贯穿的蓝色十字条纹)
    cov = np.sum(mat, axis=1) + np.sum(mat, axis=0)
    bad_bins = cov < (np.mean(cov) * mask_threshold)
    
    # 2. 计算 1D 物理期望衰减曲线 (因为单细胞稀疏性，如果不加修饰直接用，远距离全是在剧烈抖动的锯齿！)
    raw_expected_curve = np.zeros(n)
    for d in range(n):
        diag_val = np.diag(mat, d)
        valid_vals = diag_val[diag_val > 0]
        if len(valid_vals) > 0:
            raw_expected_curve[d] = np.mean(valid_vals)
        else:
            raw_expected_curve[d] = 0.0
            
    # --- 核心破局修复点 ---
    # 单细胞级别的稀疏性，让 raw_expected_curve 在大基因组距离时数值忽上忽下（比如这一条对角线平均有2个点交互，下一条突然变成0，再下一条又变成3）。
    # 如果不强行把这条一维曲线抹平，这种抖动投射到二维就会变成【大量平行于主对角线的细条纹】！！！
    smoothed_curve = gaussian_filter1d(raw_expected_curve, sigma=3.0)
    smoothed_curve[smoothed_curve < 0] = 0 # 防止平滑出负数
            
    expected = np.zeros_like(mat, dtype=float)
    for d in range(n):
        val = smoothed_curve[d]
        np.fill_diagonal(expected[d:], val)
        if d != 0:
            np.fill_diagonal(expected[:, d:], val)
            
    # 计算带有假计数平滑的 (O + p) / (E + p)
    oe_mat = (mat + pseudocount) / (expected + pseudocount)
    
    # 3. 将前面检测到的基因组死角强行涂白（值设为 1.0，因为等会取了 log2 就会变成 0，在 RdBu 中就是纯白色的缝隙，而不碍眼）
    oe_mat[bad_bins, :] = 1.0
    oe_mat[:, bad_bins] = 1.0
    
    if log2_transform:
        oe_mat = np.log2(oe_mat)
        
    return oe_mat

def plot_metacell_heatmap_enhanced(hdata, metacell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='RdBu_r', vmin=-2, vmax=2, save_path=None, dpi=300, **kwargs):
    """
    【升级增强版】带 O/E 物理背景校正的单细胞/Metacell高分辨率 Hi-C 热图可视化。
    (保留原有数据读取功能，新增 O/E 过滤消除距离衰减红晕，让TAD和核心结构更清晰)
    """
    if base_on == 'pair':
        if 'mcool' not in hdata.metacell_data or metacell_id not in hdata.metacell_data['mcool']:
            raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mcool 记录。请确保已经运行过 aggregate_metacell_pairs。")
            
        mcool_path = hdata.metacell_data['mcool'][metacell_id]
        try:
            uri = f"{mcool_path}::/resolutions/{resolution}"
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在 mcool 文件中读取分辨率 {resolution}。确保 pair 流程初始化了该分辨率。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus']:
        str_res = str(resolution)
        dict_key = base_on
        if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
            raise ValueError(f"未找到基于 {base_on} 聚合的分辨率 {resolution} 数据。请先运行对应的聚合函数。")
            
        mcool_dict = hdata.metacell_data[dict_key][str_res]
        
        # 数据类型的强健性处理：容忍用户传入 int 但字典里是 str，或反之
        if metacell_id not in mcool_dict and str(metacell_id) in mcool_dict:
            metacell_id = str(metacell_id)
        elif type(metacell_id) is str and metacell_id.isdigit() and int(metacell_id) in mcool_dict:
            metacell_id = int(metacell_id)
            
        if metacell_id not in mcool_dict:
             raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mat 聚合记录。当前内存中已有的 ID 包括: {list(mcool_dict.keys())}")
             
        if chrom not in mcool_dict[metacell_id]:
             raise ValueError(f"未找到染色体 {chrom} 的聚合矩阵。")
             
        whole_chrom_mat = mcool_dict[metacell_id][chrom]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) 
    
    # ======== 核心结构升级步骤 ========
    # 使用 O/E 矩阵校正算法替代原本的 log1p，彻底消除背景压制
    oe_mat = _calculate_oe(mat, log2_transform=True)
    oe_mat = np.nan_to_num(oe_mat)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # O/E 矩阵使用发散渐变色以 0 为中点 (红=高于预期，蓝=低于预期)
    sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
    
    ax.set_title(f"Metacell (O/E Enhanced): {metacell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return oe_mat

def plot_cell_of_metacell_heatmap_enhanced(hdata, metacell_id, cell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='RdBu_r', vmin=-2, vmax=2, save_path=None, dpi=300, **kwargs):
    """
    【升级增强版】带 O/E 物理背景校正的指定 Metacell 内单个原始细胞的高分辨率 Hi-C 热图可视化。
    """
    mat = None
    if base_on == 'pair':
        cell_name = str(cell_id)
        if isinstance(cell_id, int) or str(cell_id).isdigit():
            idx = int(cell_id)
            all_files = []
            import os
            for val in sorted(os.listdir(hdata.data_dir)):
                if val.endswith('.pairs') or val.endswith('.pairs.gz'):
                    all_files.append(val)
            if idx < len(all_files):
                cell_name = all_files[idx].split('.pairs')[0]
        
        mcool_path = None
        import os
        search_dirs = [hdata.data_dir, hdata.output_dir]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if f.endswith('.mcool') or f.endswith('.cool'):
                        if cell_name in f or str(cell_id) in f:
                            mcool_path = os.path.join(root, f)
                            break
                if mcool_path: break
            if mcool_path: break
            
        if not mcool_path:
            raise ValueError(f"未找到单细胞 '{cell_id}' 对应的 .mcool 或 .cool 文件，请确认已生成该文件或使用 base_on='mat'。")
            
        try:
            import cooler
            if mcool_path.endswith('.mcool'):
                uri = f"{mcool_path}::/resolutions/{resolution}"
            else:
                uri = f"{mcool_path}::/resolutions/{resolution}"
                if not cooler.fileops.is_multires_file(mcool_path):
                    uri = mcool_path
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在文件 {mcool_path} 中读取数据。详情: {e}")
            
    elif base_on == 'mat' or base_on == 'mat_redist':
        if not hdata.views_mat:
            raise ValueError("hdata.views_mat 为空，请确认已运行预处理。")
        if resolution not in hdata.views_mat:
            raise ValueError(f"未找到分辨率 {resolution} 的 views_mat 数据。")
        if chrom not in hdata.views_mat[resolution]:
            raise ValueError(f"未找到染色体 {chrom} 的 views_mat 数据。")
            
        try:
            cell_idx = hdata.obs.index.get_loc(cell_id)
        except KeyError:
            if isinstance(cell_id, int) and cell_id < len(hdata.obs):
                cell_idx = cell_id
            else:
                raise ValueError(f"Cell ID '{cell_id}' 不在 hdata.obs.index 中。")
                
        whole_chrom_mat = hdata.views_mat[resolution][chrom][cell_idx]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        import scipy.sparse as sp
        if sp.issparse(whole_chrom_mat):
            mat = whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
        else:
            mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) 
    oe_mat = _calculate_oe(mat, log2_transform=True)
    oe_mat = np.nan_to_num(oe_mat)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
    
    ax.set_title(f"Metacell (O/E Enhanced): {metacell_id} | Cell: {cell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return oe_mat

def plot_celltype_heatmaps_enhanced(hdata, cell_type, chrom, start, end, resolution, 
                             balance=True, base_on='pair', ncols=4, 
                             cell_type_col='cell_type', cmap='RdBu_r',
                              vmin=-2, vmax=2, save_path=None, dpi=300):
    """
    【升级增强版】可视化指定细胞类型下所有 Metacell 的 O/E 校正 Hi-C 热图 (网格展板)。
    """
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
        
    target_obs = hdata.metacells[hdata.metacells[cell_type_col] == cell_type]
    m_ids = target_obs.index.tolist()
    
    if not m_ids:
        print(f"未找到细胞类型为 '{cell_type}' 的 Metacell，请检查名称是否正确。")
        return
        
    print(f"共找到 {len(m_ids)} 个属于 '{cell_type}' 的 Metacells, 准备渲染带有物理衰减校正增强的热图...")
    
    nrows = math.ceil(len(m_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, m_id in enumerate(m_ids):
        ax = axes[i]
        ax.set_aspect('equal')
        
        try:
            if base_on == 'pair':
                if m_id not in hdata.metacell_data.get('mcool', {}):
                    ax.set_title(f"{m_id}\n(No mcool data)")
                    ax.axis('off')
                    continue
                mcool_path = hdata.metacell_data['mcool'][m_id]
                uri = f"{mcool_path}::/resolutions/{resolution}"
                clr = cooler.Cooler(uri)
                mat = clr.matrix(balance=balance).fetch((chrom, start, end))
                
            elif base_on in ['mat', 'mat_redist']:
                str_res = str(resolution)
                dict_key = 'mat' if base_on == 'mat' else 'mat_redist'
                if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
                    raise ValueError(f"未找到基于 {base_on} 聚合的分辨率数据。")
                mcool_dict = hdata.metacell_data[dict_key][str_res]
                if m_id not in mcool_dict or chrom not in mcool_dict[m_id]:
                    ax.set_title(f"{m_id}\n(No mat data)")
                    ax.axis('off')
                    continue
                
                whole_chrom_mat = mcool_dict[m_id][chrom]
                start_bin = int(start // resolution)
                end_bin = int(np.ceil(end / resolution))
                max_bins = whole_chrom_mat.shape[0]
                start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)
                mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
            else:
                raise ValueError("base_on 必须是 'pair' 或 'mat'")

            mat = np.nan_to_num(mat)
            
            # ======== 核心升级步骤 ========
            oe_mat = _calculate_oe(mat, log2_transform=True)
            oe_mat = np.nan_to_num(oe_mat)

            sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
            
            ax.set_title(m_id)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.set_title(f"{m_id}\n(Error)")
            ax.axis('off')
            
    for j in range(len(m_ids), len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"[O/E Enhanced] Cell Type: {cell_type} | Region: {chrom}:{start}-{end} @ {resolution//1000}kb", 
                 y=1.02, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()