import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        return self.purity_df_

    def get_metrics_summary(self):
        """步骤2: 打印并返回核心评估指标数值 (纯输出，不绘图)"""
        if getattr(self, 'global_score_', None) is None:
            raise RuntimeError("数据未计算！请先执行 model.calculate_metrics(cell_types)")
            
        print("-" * 40)
        print(f"简单平均纯度 (Mean Purity)  : {self.mean_purity_:.4f}")
        print(f"模型准确率 (Accuracy)      : {self.accuracy_:.4f}")
        print(f"全局加权分 (Global Score)  : {self.global_score_:.4f}")
        print("-" * 40)
        
        return {
            'mean_purity': self.mean_purity_,
            'accuracy': self.accuracy_,
            'global_score': self.global_score_
        }

    # ==========================================================
    # 以下为独立的可视化分步调用方法 (支持按需出图)
    # ==========================================================

    def plot_basic_purity(self,figsize=(8, 6)):
        """可视化 1: 基础纯度柱状图"""
        if getattr(self, 'purity_df_', None) is None:
            raise RuntimeError("数据未准备好，请先执行 model.calculate_metrics(cell_types)")
            
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=figsize)
        sns.barplot(x='CellType', y='CellType_purity', data=self.purity_df_)
        plt.xticks(rotation=45)
        plt.title('Basic Purity by Cell Type')
        plt.tight_layout()
        plt.show()

    def plot_metacell_sizes(self, figsize=(8, 6), bins=20):
        """可视化 2: Metacell 大小分布直方图"""
        if getattr(self, 'purity_df_', None) is None:
            raise RuntimeError("数据未准备好，请先执行 model.calculate_metrics(cell_types)")
            
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        plt.hist(self.purity_df_['cell_num'], bins=bins, color='skyblue', edgecolor='black')
        plt.axvline(self._avg_size_cache, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean ({self._avg_size_cache:.1f})')
        plt.axvline(self._thre_cache, color='orange', linestyle='dashed', linewidth=1.5, label=f'Threshold ({self._thre_cache:.1f})')
        plt.title('Distribution of Metacell Sizes')
        plt.xlabel('Number of Cells')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ep_score(self,figsize=(12, 6)):
        """可视化 3: 最终评估得分 EP_v2 柱状图"""
        if getattr(self, 'purity_df_', None) is None:
            raise RuntimeError("数据未准备好，请先执行 model.calculate_metrics(cell_types)")
            
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=figsize)
        sns.barplot(x='CellType', y='EP_v2', data=self.purity_df_)
        plt.xticks(rotation=45)
        plt.title('EP v2 Score by Cell Type (Corrected & Penalized)')
        plt.tight_layout()
        plt.show()

    def plot_umap_assignment(self, umap_coords,  figsize=(7,7)):
        """
        可视化 4: 单点 UMAP 散点图 (按模型分配的 Metacell ID 染色)
        注: 该图只依赖模型结果，可在 calculate_metrics 之前调用
        """
        if not hasattr(self, 'labels'):
            raise RuntimeError("未找到 self.labels，请先运行 fit() 方法完成优化。")
            
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=figsize)
        sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], 
                        hue=self.labels, palette='tab20', s=5, legend=False, rasterized=True)
        plt.title("UMAP: Metacell Assignments")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_umap_comparison(self, umap_coords,  figsize=(14, 6)):

        """
        可视化 5: 双点 UMAP 对比图 (左侧为预测平滑的类型，右侧为真实单细胞类型)
        """
        if getattr(self, '_eval_df_cache', None) is None:
            raise RuntimeError("缺少真实标签缓存，请先执行 model.calculate_metrics(cell_types)")
            
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        df = self._eval_df_cache
        
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