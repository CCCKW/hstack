import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 在这里添加或修改你想进行比较的 CSV 文件路径
# 字典的 key 会作为该技术的名字显示在图中
# ==========================================
CSV_FILES = {
    'MC2': '/Users/ckw/warehouse/metacell/stark/mc2.csv',
    'SEACELL': '/Users/ckw/warehouse/metacell/stark/seacell.csv',
    'STARK': '/Users/ckw/warehouse/metacell/stark/stark.csv'
}

def plot_funkyheatmap(csv_dict):
    metrics_means = {}
    print("Reading CSV files and computing means...")
    for name, path in csv_dict.items():
        df = pd.read_csv(path)
        # 获取除了横坐标之外的 5 个指标
        metrics = df.columns[1:]
        # 计算该技术在各个指标上的所有样本的平均表现
        means = df[metrics].mean()
        metrics_means[name] = means
        
    mean_df = pd.DataFrame(metrics_means).T

    # -------------------------------------------------------------
    # 归一化 (Min-Max Scaling) 用于映射气泡大小和颜色
    # 假设这里的指标（如 purity, acc, score）都是数值越大越好
    # -------------------------------------------------------------
    # 计算极差，为了避开全部数值一样大导致除以0，加上一个极小的值
    range_df = mean_df.max() - mean_df.min()
    range_df[range_df == 0] = 1.0  
    
    # 将每个指标缩放到 [0, 1] 之间（0表示方法中最差，1表示方法中最好）
    norm_df = (mean_df - mean_df.min()) / range_df
    
    # 综合排名：根据所有指标的归一化平局分得出（分数越高，表现越好）
    norm_df['Overall_Score'] = norm_df.mean(axis=1)
    
    # 按照综合评分从大到小排序（得分最高的排在 Dataframe 最前面）
    norm_df = norm_df.sort_values(by='Overall_Score', ascending=False)
    # 同样顺序排列原始数据，为了如果需要标注原始分数时对应得上
    mean_df = mean_df.loc[norm_df.index]
    
    methods = norm_df.index.tolist()
    metrics_cols = mean_df.columns.tolist()
    
    n_methods = len(methods)
    n_metrics = len(metrics_cols)
    
    # -------------------------------------------------------------
    # 使用 Matplotlib 开始绘制极其美观的 Funkyheatmap
    # -------------------------------------------------------------
    # 画布大小自适应
    fig, ax = plt.subplots(figsize=(n_metrics * 1.5 + 4, n_methods * 0.8 + 2.5))
    ax.set_facecolor('white')
    
    # 隐藏绘图区的所有边线和默认刻度，保持画面极端整洁
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 选用经典的渐变蓝作为点图的颜色映射
    cmap = plt.get_cmap("Blues")
    
    # 评分进度条的位置计算预设
    x_bar_start = n_metrics + 2
    bar_width_max = 2.0

    # 大标题
    ax.text(-0.5, -1.2, "Benchmarking Performance of Metacell Methods", 
            fontsize=16, fontweight='bold', ha='left', color='#222222')
    
    # 顶部的列名字段
    for j, metric in enumerate(metrics_cols):
        ax.text(j + 1, -0.5, metric, ha='center', va='bottom', 
                rotation=30, fontsize=11, fontweight='bold', color='#555555')
        
    ax.text(x_bar_start + bar_width_max/2, -0.5, 'Overall Score\n(Normalized)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#555555')

    # 按排名顺序，从第一名往下绘制每一行
    for i, method in enumerate(methods):
        y = i
        
        # 1. 绘制行的底层交替背景 (增加横向阅读的视觉引导)
        if i % 2 == 0:
            rect = patches.Rectangle((-1.5, y - 0.5), n_metrics + bar_width_max + 4.5, 1, 
                                     linewidth=0, color='#f7f9fc', zorder=0)
            ax.add_patch(rect)
            
        # 2. 最左侧绘制排名与分析技术名称
        ax.text(-0.5, y, method, ha='right', va='center', fontsize=12, fontweight='bold', color='#333333')
        
        # 3. 绘制各种指标数据的气泡矩阵
        for j, metric in enumerate(metrics_cols):
            x = j + 1
            val = norm_df.loc[method, metric] # 范围在 [0, 1] 之间
            
            # 画一个带有边缘线的底层“凹槽辅助圆”，用于在数值极差导致气泡极小时凸显轮廓
            ax.scatter(x, y, s=500, color='white', edgecolor='#eaeaea', zorder=1)
            
            # 真实代表数值的气泡（颜色和面积都代表分数）
            # 颜色截断（0.2-1.0），防止 0 分呈现出不可见的纯透明背景白
            color = cmap(0.2 + 0.8 * val)
            # 气泡面积范围: 最差的给 30 base 面积，最好的给 480 面积
            size = 30 + 450 * val 
            ax.scatter(x, y, s=size, color=color, edgecolor='none', zorder=2)
            
        # 4. 最右侧绘制整体综合评分 (Overall Score) 形似长方形进度条
        overall_val = norm_df.loc[method, 'Overall_Score']
        
        # 底层浅灰色的背景轨
        ax.barh(y, bar_width_max, left=x_bar_start, height=0.35, color='#eeeeee', zorder=1)
        # 真实红橙色进度条代表实际综合得分
        ax.barh(y, overall_val * bar_width_max, left=x_bar_start, height=0.35, color='#f26b5b', zorder=2)
        
        # 进度条末尾打印出具体的综合分数 (比如 0.85)
        ax.text(x_bar_start + overall_val * bar_width_max + 0.08, y, 
                f"{overall_val:.2f}", va='center', fontsize=10, 
                color='#333333', fontweight='heavy')

    # 将 Y 轴反转，这样 y=0（得分最高的方法）就会绘制在图表的最顶端
    ax.set_ylim(-1.5, n_methods)
    ax.invert_yaxis()
    
    plt.tight_layout()
    print("Funkyheatmap rendered successfully! Displaying window...")
    
    # =====================================
    # 如果想保存这幅精美的高清科研图，可以取消注释
    # plt.savefig('funkyheatmap_ranking.pdf', bbox_inches='tight') 
    # plt.savefig('funkyheatmap_ranking.png', dpi=300, bbox_inches='tight')
    # =====================================
    plt.show()

if __name__ == "__main__":
    plot_funkyheatmap(CSV_FILES)
