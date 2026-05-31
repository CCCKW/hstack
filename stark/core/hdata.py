import pandas as pd
import numpy as np
import os

# 支持的模态类型
MODALITY_HIC  = "hic"
MODALITY_RNA  = "rna"
MODALITY_METH = "meth"
MODALITY_ATAC = "atac"


def _make_view_key(modality: str, resolution: int = None) -> str:
    """
    生成标准化的视图键。
    - HiC 带分辨率: "hic_50000"
    - 其他组学无需分辨率: "rna", "meth"
    - 若想给同组学区分: "rna_sample1"
    """
    if resolution is not None:
        return f"{modality}_{resolution}"
    return modality


class HData:
    """
    Stark 多级数据容器。
    支持多组学多视图 (Hi-C 多分辨率、RNA、METH、ATAC 等)。

    视图键 (view key) 规范:
        - HiC:  "hic_<分辨率>"  例如 "hic_50000"
        - RNA:  "rna"  或带后缀 "rna_sample1"
        - METH: "meth" 或带分辨率 "meth_50000"
        - 兼容旧接口: 整数键 50000 => 等价于 "hic_50000"

    视图配置存储在 hdata.view_configs:
        {
          "hic_50000": {"modality": "hic", "resolution": 50000, "data_dir": ...},
          "rna":       {"modality": "rna", "resolution": None,  "data_dir": ...},
        }
    """
    def __init__(self, data_dir=None, output_dir=None, genome_reference_path=None,
                 chrom_list=None, resolutions=None):
        # --- 基础配置 (Hi-C 主路径) ---
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.genome_reference_path = genome_reference_path
        self.chrom_list = chrom_list if chrom_list is not None else []
        # 兼容旧 h5ad：resolutions 可能被序列化为字符串列表
        self.resolutions = sorted(int(r) for r in resolutions) if resolutions else []

        # --- 视图元信息 ---
        # key: view_key (str), value: dict with modality/resolution/data_dir/...
        self.view_configs = {}
        # 自动为每个 HiC 分辨率注册视图配置
        for res in self.resolutions:
            vk = _make_view_key(MODALITY_HIC, res)
            self.view_configs[vk] = {
                "modality": MODALITY_HIC,
                "resolution": res,
                "data_dir": data_dir,
            }

        # --- 单细胞层次数据 (键为 view_key 字符串) ---
        self.views_pca = {}
        self.views_umap = {}
        self.views_embedding = {}
        self.views_mat = {}
        self.views_is = {}
        self.obs = pd.DataFrame()

        # --- Metacell 层次数据 ---
        self.metacells = pd.DataFrame()
        self.metacell_data = {
            'pairs': {},
            'cool': {},
            'mcool': {}
        }

        # --- 状态与模型 ---
        self.uns = {}
        self.model = None

    # ------------------------------------------------------------------
    # 视图管理 API
    # ------------------------------------------------------------------
    def add_view_config(self, view_key: str, modality: str, resolution: int = None,
                        data_dir: str = None, **kwargs):
        """
        注册一个新视图的元信息。
        view_key 建议用 _make_view_key(modality, resolution) 生成。

        示例:
            hdata.add_view_config("rna", modality="rna", data_dir="/path/to/rna.h5ad")
            hdata.add_view_config("meth_50000", modality="meth", resolution=50000,
                                  data_dir="/path/to/meth")
        """
        self.view_configs[view_key] = {
            "modality": modality,
            "resolution": resolution,
            "data_dir": data_dir,
            **kwargs,
        }

    def add_view(self, view_key: str, modality: str,
                 pca=None, umap=None, embedding=None, mat=None, is_score=None,
                 resolution: int = None, data_dir: str = None, **kwargs):
        """
        一次性注册视图配置并写入对应的 views_* 数据。

        参数:
            view_key:   视图唯一标识字符串，如 "rna", "meth_50000"
            modality:   模态类型，建议用常量 MODALITY_HIC/RNA/METH/ATAC
            pca:        (N, d) ndarray，PCA 降维结果
            umap:       (N, 2) ndarray，UMAP 坐标
            embedding:  (N, d) ndarray，原始嵌入向量
            mat:        dict {chrom: sparse_matrix}，稀疏接触矩阵 (HiC 专用)
            is_score:   (N, bins) ndarray，绝缘分数 (HiC 专用)
            resolution: 分辨率 (HiC/METH 有意义)
            data_dir:   该组学的原始数据路径
        """
        self.add_view_config(view_key, modality=modality, resolution=resolution,
                             data_dir=data_dir, **kwargs)
        if pca is not None:
            self.views_pca[view_key] = pca
        if umap is not None:
            self.views_umap[view_key] = umap
        if embedding is not None:
            self.views_embedding[view_key] = embedding
        if mat is not None:
            self.views_mat[view_key] = mat
        if is_score is not None:
            self.views_is[view_key] = is_score

    def list_views(self):
        """返回所有已注册视图的摘要 DataFrame。"""
        rows = []
        all_keys = set(self.view_configs) | set(self.views_pca) | \
                   set(self.views_umap) | set(self.views_embedding) | \
                   set(self.views_mat) | set(self.views_is)
        for vk in sorted(all_keys, key=str):
            cfg = self.view_configs.get(vk, {})
            rows.append({
                "view_key":  vk,
                "modality":  cfg.get("modality", "unknown"),
                "resolution": cfg.get("resolution"),
                "has_pca":   vk in self.views_pca,
                "has_umap":  vk in self.views_umap,
                "has_embedding": vk in self.views_embedding,
                "has_mat":   vk in self.views_mat,
                "has_is":    vk in self.views_is,
            })
        return pd.DataFrame(rows).set_index("view_key") if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # 向后兼容: 用 int 分辨率访问 == "hic_<res>"
    # ------------------------------------------------------------------
    def _resolve_view_key(self, key):
        """将 int 键 (旧接口) 自动转换为 "hic_<res>" 字符串键。"""
        if isinstance(key, int):
            return _make_view_key(MODALITY_HIC, key)
        return key

    @property
    def n_cells(self):
        return len(self.obs) if not self.obs.empty else 0

    @property
    def n_metacells(self):
        return len(self.metacells) if not self.metacells.empty else 0

    def __repr__(self):
        descr = f"HData object with {self.n_cells} cells and {self.n_metacells} metacells\n"
        descr += f"    obs columns: {list(self.obs.columns)}\n"

        # 分组展示多视图信息
        if self.view_configs:
            descr += f"    views ({len(self.view_configs)} registered):\n"
            for vk, cfg in self.view_configs.items():
                mod = cfg.get("modality", "?")
                res = cfg.get("resolution")
                res_str = f"  res={res}" if res is not None else ""
                has = []
                if vk in self.views_pca:       has.append("pca")
                if vk in self.views_umap:      has.append("umap")
                if vk in self.views_embedding: has.append("emb")
                if vk in self.views_mat:       has.append("mat")
                if vk in self.views_is:        has.append("is")
                descr += f"      [{vk}] modality={mod}{res_str}  data={has}\n"
        else:
            descr += f"    views_pca: {list(self.views_pca.keys())}\n"

        descr += f"    uns keys: {list(self.uns.keys())}\n"

        if self.n_metacells > 0:
            descr += f"    metacells cols: {list(self.metacells.columns)}\n"
            data_types = [k for k, v in self.metacell_data.items() if v]
            descr += f"    metacell_data: {data_types}\n"

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
        # 序列化多视图元信息
        adata.uns['view_configs'] = _stringify_keys(getattr(self, 'view_configs', {}))

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
            resolutions=[int(r) for r in base_attrs.get('resolutions', [])]
        )
        
        obj.metacell_data = _destringify_keys(adata.uns.get('metacell_data', {'pairs': {}, 'cool': {}, 'mcool': {}}))
        obj.obs = adata.obs
        if 'metacells' in adata.uns:
             obj.metacells = adata.uns['metacells']

        # 恢复 view_configs (新格式直接读取，无需 destringify)
        raw_view_configs = adata.uns.get('view_configs', {})
        obj.view_configs = dict(raw_view_configs)

        # Ensure all view dicts exist
        for prefix in ['pca', 'umap', 'embedding', 'mat', 'is']:
            if not hasattr(obj, f"views_{prefix}"):
                setattr(obj, f"views_{prefix}", {})

        def _migrate_view_key(view_k: str) -> str:
            """将旧的纯数字字符串键 (e.g. "50000") 迁移为新格式 "hic_50000"。"""
            if view_k.isdigit():
                new_k = _make_view_key(MODALITY_HIC, int(view_k))
                # 若 view_configs 中没有对应记录，补充一条
                if new_k not in obj.view_configs:
                    obj.view_configs[new_k] = {
                        "modality": MODALITY_HIC,
                        "resolution": int(view_k),
                        "data_dir": obj.data_dir,
                    }
                return new_k
            return view_k

        # Restore views from obsm
        for k in list(adata.obsm.keys()):
            if k.startswith("views_"):
                parts = k.split("_", 2)
                if len(parts) == 3:
                    view_k = _migrate_view_key(parts[2])
                    getattr(obj, f"views_{parts[1]}")[view_k] = adata.obsm[k]

        for k, v in adata.uns.items():
            if k.startswith("uns_"):
                obj.uns[k.replace("uns_", "", 1)] = v
            elif k.startswith("__failed_views_"):
                parts = k.replace("__failed_views_", "", 1).split("_", 1)
                if len(parts) == 2:
                    prefix, view_k = parts
                    view_k = _migrate_view_key(view_k)
                    getattr(obj, f"views_{prefix}")[view_k] = pickle.loads(
                        v.tobytes() if hasattr(v, "tobytes") else bytes(v))

        model_bytes = adata.uns.get('hdata_model_bytes', None)
        if model_bytes is not None:
             obj.model = pickle.loads(model_bytes.tobytes() if hasattr(model_bytes, "tobytes") else bytes(model_bytes))

        return obj