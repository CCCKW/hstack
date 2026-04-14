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
        self.resolutions = sorted(resolutions) 
        
        # --- 单细胞层次数据 ---
        self.views_pca = {}
        self.views_umap = {}
        self.views_embedding = {}
        self.views_mat = {}
        self.views_is = {}
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
        if hasattr(self, 'views_mat'):
            descr += f"    views_mat: {list(self.views_mat.keys())}\n"
        if hasattr(self, 'views_is'):
            descr += f"    views_is: {list(self.views_is.keys())}\n"
        descr += f"    uns keys: {list(self.uns.keys())}\n"
        
        # 打印 Metacell 信息
        if self.n_metacells > 0:
            descr += f"    metacells: {list(self.metacells.columns)}\n"
            data_types = [k for k, v in self.metacell_data.items() if v]
            descr += f"    metacell_data keys: {data_types}\n"
            
        if self.model is not None:
            descr += f"    model: MultiViewSEACells (trained: {getattr(self.model, 'initialized', False)})\n"
        return descr

    def write_h5ad(self, path):
        """
        以类似 scanpy/anndata 各视图信息导出为 .h5ad 格式。
        基本视图特征会尽可能保存在 AnnData.obsm 中；不支持的特殊结构会以 bytes (pickle格式) 保存在 uns 中。
        包含训练过的模型。
        """
        import anndata as ad
        import pickle
        import numpy as np

        obs = self.obs.copy() if (hasattr(self, 'obs') and not self.obs.empty) else pd.DataFrame(index=[str(i) for i in range(self.n_cells)])
        if not obs.empty:
            obs.index = obs.index.astype(str)

        adata = ad.AnnData(obs=obs)

        # Store unstructured and metadata
        def _stringify_keys(d):
            if isinstance(d, dict):
                return {str(k): _stringify_keys(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [_stringify_keys(i) for i in d]
            return d

        def _destringify_keys(d):
            if isinstance(d, dict):
                res = {}
                for k, v in d.items():
                    if isinstance(k, str) and k.isdigit():
                        k = int(k)
                    res[k] = _destringify_keys(v)
                return res
            elif isinstance(d, list):
                return [_destringify_keys(i) for i in d]
            return d

        adata.uns['hdata_base_attrs'] = _stringify_keys({
            'data_dir': getattr(self, 'data_dir', None),
            'output_dir': getattr(self, 'output_dir', None),
            'genome_reference_path': getattr(self, 'genome_reference_path', None),
            'chrom_list': getattr(self, 'chrom_list', []),
            'resolutions': getattr(self, 'resolutions', [])
        })
        adata.uns['metacell_data'] = _stringify_keys(getattr(self, 'metacell_data', {'pairs': {}, 'cool': {}, 'mcool': {}}))
        
        if hasattr(self, 'metacells') and not self.metacells.empty:
            adata.uns['metacells'] = self.metacells

        if hasattr(self, 'uns') and isinstance(self.uns, dict):
            for k, v in self.uns.items():
                adata.uns[f"uns_{k}"] = _stringify_keys(v)

        # Store views dynamically. It can natively go to obsm if it is ndarray or sparse.
        # Otherwise, fall back to pickle in numpy uint8 byte arrays to store arbitrary dict elements.
        for prefix in ['pca', 'umap', 'embedding', 'mat', 'is']:
            view_dict = getattr(self, f"views_{prefix}", {})
            for k, v in view_dict.items():
                try:
                    # AnnData requirement: First dimension of obsm has to match n_obs
                    # and must be an array-like or sparse matrix.
                    if getattr(v, "shape", [0])[0] == self.n_cells:
                         adata.obsm[f"views_{prefix}_{k}"] = v
                    else:
                         raise ValueError("Shape mismatch")
                except Exception:
                    # Pickling to uint8 array to store any unsupported objects natively in h5ad.
                    adata.uns[f"__failed_views_{prefix}_{k}"] = np.frombuffer(pickle.dumps(v), dtype=np.uint8)

        # Store the model
        if getattr(self, 'model', None) is not None:
             adata.uns['hdata_model_bytes'] = np.frombuffer(pickle.dumps(self.model), dtype=np.uint8)
        else:
             adata.uns['hdata_model_bytes'] = None

        adata.write_h5ad(path)

    @classmethod
    def read_h5ad(cls, path):
        """
        从 .h5ad 恢复完整的 HData 对象 (对应自带 write_h5ad 的逆反过程)。
        """
        import anndata as ad
        import pickle

        def _destringify_keys(d):
            if isinstance(d, dict):
                res = {}
                for k, v in d.items():
                    if isinstance(k, str) and k.isdigit():
                        k = int(k)
                    res[k] = _destringify_keys(v)
                return res
            elif isinstance(d, list):
                return [_destringify_keys(i) for i in d]
            return d

        adata = ad.read_h5ad(path)
        base_attrs = _destringify_keys(adata.uns.get('hdata_base_attrs', {}))
        
        obj = cls(
            data_dir=base_attrs.get('data_dir'),
            output_dir=base_attrs.get('output_dir'),
            genome_reference_path=base_attrs.get('genome_reference_path'),
            chrom_list=base_attrs.get('chrom_list'),
            resolutions=base_attrs.get('resolutions', [])
        )
        
        obj.metacell_data = _destringify_keys(adata.uns.get('metacell_data', {'pairs': {}, 'cool': {}, 'mcool': {}}))
        obj.obs = adata.obs
        if 'metacells' in adata.uns:
             obj.metacells = adata.uns['metacells']

        # Ensure all view dicts exist
        for prefix in ['pca', 'umap', 'embedding', 'mat', 'is']:
            if not hasattr(obj, f"views_{prefix}"):
                setattr(obj, f"views_{prefix}", {})
                
        # Restore unstructured and views
        for k in list(adata.obsm.keys()):
            if k.startswith("views_"):
                parts = k.split("_", 2)
                if len(parts) == 3:
                     view_k = parts[2]
                     if isinstance(view_k, str) and view_k.isdigit():
                         view_k = int(view_k)
                     getattr(obj, f"views_{parts[1]}")[view_k] = adata.obsm[k]
                     
        for k, v in adata.uns.items():
            if k.startswith("uns_"):
                obj.uns[k.replace("uns_", "", 1)] = v
            elif k.startswith("__failed_views_"):
                parts = k.replace("__failed_views_", "", 1).split("_", 1)
                if len(parts) == 2:
                     prefix, view_k = parts
                     if isinstance(view_k, str) and view_k.isdigit():
                         view_k = int(view_k)
                     getattr(obj, f"views_{prefix}")[view_k] = pickle.loads(v.tobytes() if hasattr(v, "tobytes") else bytes(v))

        model_bytes = adata.uns.get('hdata_model_bytes', None)
        if model_bytes is not None:
             obj.model = pickle.loads(model_bytes.tobytes() if hasattr(model_bytes, "tobytes") else bytes(model_bytes))

        return obj