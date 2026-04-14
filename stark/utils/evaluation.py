import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

class EvaluationMixin:
    """提供模型综合评估和对应可视化功能的 Mixin 类 (完全解耦，支持分步调用)"""

    def calculate_metrics(self, cell_types):
        """
        步骤1: 计算核心评估指标，并将其缓存为模型属性 (纯计算，不绘图)
        
        参数:
        - cell_types: array-like, 细胞的真实类型标签
        
        返回:
        - purity_df: 包含各项纯度和大小评估指标的 DataFrame
        """
        if not hasattr(self, 'labels'):
            raise RuntimeError("未找到 self.labels，请先运行 fit() 方法完成优化。")
            
        print("\n" + "=" * 60)
        print("正在计算评估指标...")
        print("=" * 60)

        # 整理基础 DataFrame
        df = pd.DataFrame({'CellType': cell_types, 'Metacell': self.labels})
        
        over_merge = calculate_overmerging_metrics(df['Metacell'], df['CellType'])
        
        wcos = over_merge['WCOS']
        hwis = over_merge['HWIS']
        
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
        self.mean_purity_ = purity['CellType_purity'].mean()
        self.global_score_ = (purity['EP_v2'] * purity['cell_num']).sum() / purity['cell_num'].sum()
        
        # 计算 Accuracy 并映射标签以便画图
        hash_meta = purity['CellType'].to_dict()
        df['meta_lb'] = df['Metacell'].map(hash_meta)
        self.accuracy_ = (df['CellType'] == df['meta_lb']).sum() / df.shape[0]

        # 保存结果及画图必需的缓存
        self.purity_df_ = purity
        self._eval_df_cache = df
        self._avg_size_cache = avg_size
        self._thre_cache = thre

        print(f"✅ 指标计算完成！(发现 {num_unique_types} 种细胞类型)")
        return self.purity_df_, self._eval_df_cache, self._avg_size_cache, self._thre_cache

    def get_metrics_summary(self):
        """步骤2: 打印并返回核心评估指标数值 (纯输出，不绘图)"""

        print("-" * 40)
        print(f"简单平均纯度 (Mean Purity)  : {self.mean_purity_:.4f}")
        print(f"模型准确率 (Accuracy)      : {self.accuracy_:.4f}")
        print(f"全局加权分 (Global Score)  : {self.global_score_:.4f}")
        print()
        print("-" * 40)
        
        return {
            'mean_purity': self.mean_purity_,
            'accuracy': self.accuracy_,
            'global_score': self.global_score_
        }

    # ==========================================================
    # 以下为独立的可视化分步调用方法 (支持按需出图)
    # ==========================================================
