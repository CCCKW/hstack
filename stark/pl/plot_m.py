import math
import cooler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

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