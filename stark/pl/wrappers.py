import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
# ==========================================================
# 1. 探索性数据可视化 (复刻自您原有的 pipe.py)
# ==========================================================
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

def plot_metacells(hdata, ref_view_res=500000, max_size=500, min_size=100, show_idx=True):
    """可视化得到的 Metacell 的位置和大小分布"""
    if hdata.model is None:
        raise ValueError("模型尚未初始化或拟合。")
    hdata.model.plot_metacells(
        hdata.views_umap[ref_view_res], 
        title="Final Metacell Positions", 
        show_idx=show_idx, 
        max_size=max_size, 
        min_size=min_size
    )

def plot_specific_metacell(hdata, idx, ref_view_res=500000):
    """诊断性可视化：查看单一指定的 Metacell 及其覆盖范围"""
    if hdata.model is None:
        raise ValueError("模型尚未初始化或拟合。")
    hdata.model.plot_specific_metacell(hdata.views_umap[ref_view_res], idx)

# ==========================================================
# 3. 评估指标可视化 (代理底层 evaluation.py 中的出图方法)
# ==========================================================

def plot_basic_purity(hdata, figsize=(8, 6)):
    """纯度柱状图"""
    if hdata.model is None or not hasattr(hdata.model, 'purity_df_'):
        raise ValueError("请先运行 sk.tl.evaluate(hdata) 计算指标")
    hdata.model.plot_basic_purity(figsize=figsize)

def plot_metacell_sizes(hdata, figsize=(8, 6), bins=20):
    """Metacell 大小分布检查"""
    if hdata.model is None or not hasattr(hdata.model, 'purity_df_'):
        raise ValueError("请先运行 sk.tl.evaluate(hdata) 计算指标")
    hdata.model.plot_metacell_sizes(figsize=figsize, bins=bins)

def plot_ep_score(hdata, figsize=(12, 6)):
    """展示最终 EP_v2 综合打分柱状图"""
    if hdata.model is None or not hasattr(hdata.model, 'purity_df_'):
        raise ValueError("请先运行 sk.tl.evaluate(hdata) 计算指标")
    hdata.model.plot_ep_score(figsize=figsize)

def plot_umap_assignment(hdata, ref_view_res=500000, figsize=(7, 7)):
    """展示由模型分配的 Metacell ID 染色的散点图"""
    if hdata.model is None:
        raise ValueError("模型尚未拟合。")
    hdata.model.plot_umap_assignment(hdata.views_umap[ref_view_res], figsize=figsize)

def plot_umap_comparison(hdata, ref_view_res=500000, figsize=(14, 6)):
    """可视化最后的结果：左侧平滑预测类别 vs 右侧原始类别"""
    if hdata.model is None or not hasattr(hdata.model, '_eval_df_cache'):
        raise ValueError("请先运行 sk.tl.evaluate(hdata) 生成画图缓存")
    hdata.model.plot_umap_comparison(hdata.views_umap[ref_view_res], figsize=figsize)