import pandas as pd
import numpy as np
import os

class HData:
    """
    Stark 多级数据容器。
    不仅存储单细胞层次的数据 (obs, views)，还存储 Metacell 层次的数据 (metacells)。
    """
    def __init__(self, data_dir=None, output_dir=None, genome_reference_path=None, chrom_list=None, resolutions=None):
        # --- 基础配置 ---
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.genome_reference_path = genome_reference_path
        self.chrom_list = chrom_list if chrom_list is not None else []
        self.resolutions = resolutions if resolutions is not None else []
        
        # --- 单细胞层次数据 ---
        self.views_pca = {}
        self.views_umap = {}
        self.views_embedding = {}
        self.obs = pd.DataFrame() 
        
        # --- Metacell 层次数据 (新增) ---
        # metacells 存储 Metacell 的元数据 (每一行是一个 Metacell)
        # 比如: 总深度, 包含细胞数, 优势细胞类型等
        self.metacells = pd.DataFrame() 
        
        # metacell_data 存储生成的重度文件路径映射
        # 结构: {'pairs': {id: path}, 'cool': {res: {id: path}}, 'mcool': {id: path}}
        self.metacell_data = {
            'pairs': {},
            'cool': {},
            'mcool': {}
        }
        
        # --- 状态与模型 ---
        self.uns = {}
        self.model = None

    @property
    def n_cells(self):
        return len(self.obs) if not self.obs.empty else 0

    @property
    def n_metacells(self):
        return len(self.metacells) if not self.metacells.empty else 0

    def __repr__(self):
        descr = f"HData object with {self.n_cells} cells and {self.n_metacells} metacells\n"
        descr += f"    resolutions: {self.resolutions}\n"
        descr += f"    obs: {list(self.obs.columns)}\n"
        descr += f"    views_pca: {list(self.views_pca.keys())}\n"
        descr += f"    views_umap: {list(self.views_umap.keys())}\n"
        descr += f"    views_embedding: {list(self.views_embedding.keys())}\n"
        descr += f"    uns keys: {list(self.uns.keys())}\n"
        
        # 打印 Metacell 信息
        if self.n_metacells > 0:
            descr += f"    metacells: {list(self.metacells.columns)}\n"
            data_types = [k for k, v in self.metacell_data.items() if v]
            descr += f"    metacell_data keys: {data_types}\n"
            
        if self.model is not None:
            descr += f"    model: MultiViewSEACells (trained: {self.model.initialized})\n"
        return descr