import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 只需要在这里替换为你想要读取的 CSV 文件路径
# ==========================================
CSV_PATH = '/Users/ckw/warehouse/metacell/stark/seacell.csv'

def plot_robustness(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 第一列没有列名（通常会被 Pandas 读为 'Unnamed: 0'），我们将其作为横坐标（metacell 个数）
    x_col = df.columns[0]
    
    # 剩下的全部列就是指标（5个指标）
    metrics = df.columns[1:]
    
    plt.figure(figsize=(10, 6))
    
    # 定义一些标记形状，让曲线区分度更高
    markers = ['o', 's', '^', 'D', 'v']
    
    # 遍历绘制 5 个指标的折线图
    for i, metric in enumerate(metrics):
        plt.plot(df[x_col], df[metric], 
                 marker=markers[i % len(markers)], 
                 label=metric, 
                 linewidth=2)
    
    # 标题使用具体的文件名作为标识
    tech_name = csv_path.split('/')[-1].replace('.csv', '').upper()
    plt.title(f"Robustness of Metrics - {tech_name}", fontsize=14, fontweight='bold')
    
    plt.xlabel('Number of Metacells', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # 开启网格线，有助于观察数值变化
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 将图例放到右上角或最适合的位置
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 如需保存图片到本地，可以取消下方注释
    # plt.savefig(f'{tech_name}_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_robustness(CSV_PATH)
