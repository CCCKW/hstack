from ..utils.rec_num import recommend_by_leiden
from ..utils.model import MultiViewSEACells
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from stark.core.hdata import _make_view_key, MODALITY_HIC


def _resolve_pca_view(hdata, key):
    """
    将 int 分辨率键或字符串键统一解析为 views_pca 里实际存在的键。
    - int 500000  -> 先尝试 'hic_500000'，再回退到直接 int
    - str 'rna'   -> 直接用
    """
    if isinstance(key, int):
        str_key = _make_view_key(MODALITY_HIC, key)
        if str_key in hdata.views_pca:
            return str_key
        # 向后兼容：万一仍是 int 键
        if key in hdata.views_pca:
            return key
        raise KeyError(f"views_pca 中未找到视图 '{str_key}'（也没有旧式 int 键 {key}），"
                       f"现有键: {list(hdata.views_pca.keys())}")
    if key in hdata.views_pca:
        return key
    raise KeyError(f"views_pca 中未找到视图 '{key}'，现有键: {list(hdata.views_pca.keys())}")


def evaluate_metacell(hdata, use_view=None, metric="euclidean"):
    """
    步骤 X: 评估 Metacell 的空间分布特征，计算紧凑度 (Compactness) 和 分离度 (Separation)。

    参数:
    hdata: 核心数据对象，需包含降维特征 (hdata.views_pca) 和 metacell 分配标签 (hdata.obs['metacell'])
    use_view (int/str): 指定使用哪个视角的降维特征。如果为 None，则默认使用第一个视角。
    metric (str): 距离度量方式，默认 'euclidean' (欧氏距离)，也可设为 'cosine' (余弦距离)。

    返回:
    res_df (pd.DataFrame): 包含每个 metacell 详细指标的表格。
    metrics_summary (dict): 包含全局平均紧凑度和分离度的字典。
    """
    print("\n" + "=" * 60)
    print("正在计算 Metacell 空间结构指标 (Compactness & Separation)...")
    print("=" * 60)

    # 1. 尝试获取降维坐标矩阵
    if use_view is None:
        used_view_name = list(hdata.views_pca.keys())[0]
    else:
        used_view_name = _resolve_pca_view(hdata, use_view)
    coords = hdata.views_pca[used_view_name]

    print(f"使用的特征视图: {used_view_name} (距离度量: {metric})")

    # 2. 拿到已有分配结果
    cell_to_metacell = hdata.obs["metacell"].values
    df = pd.DataFrame({"metacell_id": cell_to_metacell})

    mc_ids = []
    compactness_list = []
    centroids = []

    # 3. 计算每个 metacell 的组内距 (紧凑度) 和 质心坐标
    groups = df.groupby("metacell_id")
    for mc_id, group in groups:
        idx = group.index.values
        sub_coords = coords[idx]

        # 针对局部聚类群，取平均求得质心
        centroid = sub_coords.mean(axis=0)

        if len(sub_coords) > 1:
            dists = pairwise_distances(
                sub_coords, centroid.reshape(1, -1), metric=metric
            )
            compactness = dists.mean()
        else:
            compactness = 0.0  # 仅一个细胞时设为 0

        mc_ids.append(mc_id)
        compactness_list.append(compactness)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # 4. 计算不同质心之间的最近距离 (分离度)
    if len(centroids) > 1:
        cent_dists = pairwise_distances(centroids, metric=metric)
        # 排除自身到自身距离带来的干扰
        np.fill_diagonal(cent_dists, np.inf)
        separation_list = cent_dists.min(axis=1)
    else:
        separation_list = np.array([0.0])

    # 5. Min-Max 归一化到 [0, 1] 之间，并统一格式化为“得分 (Score)：1为最好，0为最差”
    comp_min, comp_max = np.min(compactness_list), np.max(compactness_list)
    sep_min, sep_max = np.min(separation_list), np.max(separation_list)

    comp_range = (comp_max - comp_min) if (comp_max - comp_min) > 0 else 1.0
    sep_range = (sep_max - sep_min) if (sep_max - sep_min) > 0 else 1.0

    # 原始距离：越小越紧凑 -> 也就是距离越短得分越高
    compactness_score = 1.0 - (np.array(compactness_list) - comp_min) / comp_range
    # 原始距离：越大越分离 -> 就是距离越远得分越高
    separation_score = (np.array(separation_list) - sep_min) / sep_range

    # 构建结果 DataFrame
    res_df = pd.DataFrame(
        {
            "raw_compactness_dist": compactness_list,
            "raw_separation_dist": separation_list,
            "compactness": compactness_score,  # [0, 1] 得分，1 最优
            "separation": separation_score,  # [0, 1] 得分，1 最优
        },
        index=mc_ids,
    )

    mean_compactness = float(res_df["compactness"].mean())
    mean_separation = float(res_df["separation"].mean())

    # ====== 仿照 evaluate 函数实现格式化输出打印 ======
    print("-" * 40)
    print(
        f"全局平均紧凑度得分 (Compactness Score): {mean_compactness:.4f} (值域 0~1，1=最紧凑/最佳)"
    )
    print(
        f"全局平均分离度得分 (Separation Score) : {mean_separation:.4f} (值域 0~1，1=最分离/最佳)"
    )
    print("-" * 40)

    metrics_summary = {
        "mean_compactness": mean_compactness,
        "mean_separation": mean_separation,
    }

    # 6. 将评测结果向基础对象属性中打入
    if "metrics" not in hdata.uns:
        hdata.uns["metrics"] = {}
    hdata.uns["metrics"].update(metrics_summary)

    hdata.uns["cs_res_df"] = res_df
    hdata.uns["mean_compactness"] = mean_compactness
    hdata.uns["mean_separation"] = mean_separation

    # 7. 追加到主元明细表 hdata.metacells
    if (
        not hasattr(hdata, "metacells")
        or hdata.metacells is None
        or hdata.metacells.empty
    ):
        hdata.metacells = res_df.copy()
    else:
        # 针对 hdata.metacells，通过索引对其进行映射插入，以免不同步骤导致的排序差异
        hdata.metacells["raw_compactness_dist"] = res_df["raw_compactness_dist"]
        hdata.metacells["raw_separation_dist"] = res_df["raw_separation_dist"]
        hdata.metacells["compactness"] = res_df["compactness"]
        hdata.metacells["separation"] = res_df["separation"]

    print("✅ 空间分布评估指标计算完成，得已并入 hdata.metacells 及 hdata.uns 缓存中。")
    return res_df, metrics_summary


def calculate_overmerging_metrics(cell_to_metacell, true_labels):
    """
    评估 Metacell 划分中是否出现“巨大且不纯”的过度融合现象 (Over-merging / Hub effect)。

    参数:
    cell_to_metacell (array-like): 长度为 N 的数组，记录每个单细胞被分配到的 metacell ID。
    true_labels (array-like): 长度为 N 的数组，记录每个单细胞的真实细胞类型 (Ground truth)。

    返回:
    dict: 包含 WCOS 和 HWIS 两个 [0, 1] 之间的指标。值越接近 1 越好，越接近 0 问题越严重。
    """
    df = pd.DataFrame({"metacell_id": cell_to_metacell, "true_label": true_labels})

    total_cells = len(df)

    # 用于记录每个 metacell 的惩罚值
    penalties = []
    squared_penalties = []

    # 遍历每一个 metacell
    for mc_id, group in df.groupby("metacell_id"):
        size = len(group)
        size_fraction = size / total_cells  # 该 metacell 占总数据的比例 (0 到 1)

        # 计算该 metacell 的纯度 (Purity)
        # 纯度 = 该 metacell 中数量最多的真实细胞类型的占比
        mode_count = group["true_label"].value_counts().iloc[0]
        purity = mode_count / size
        impurity = 1.0 - purity  # 不纯度 (0 到 1)

        # 计算惩罚项
        penalty = size_fraction * impurity
        squared_penalty = (size_fraction**2) * impurity

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

    return {"WCOS": float(wcos), "HWIS": float(hwis)}


def recommend_metacell_num(
    hdata,
    target_depth_min=20 * 1e6,
    target_depth_max=40 * 1e6,
    resolution_param=2.0,
    n_neighbors=15,
    ref_view=1000000,
    plot_result=True,
    save_path = None
):
    """
    步骤 2: 推荐 MetaCell 范围并保存到 hdata
    完全调用您底层的 recommend_by_leiden，不改变任何算法逻辑。
    """
    # 从 HData 的统一表格中抽取 depth
    depth_array = hdata.obs["depth"].values

    min_k, max_k = recommend_by_leiden(
        depth_array=depth_array,
        features=hdata.views_pca[_resolve_pca_view(hdata, ref_view)],
        target_depth_min=target_depth_min,
        target_depth_max=target_depth_max,
        resolution=resolution_param,
        n_neighbors=n_neighbors,
        plot_result=plot_result,
        save_path=save_path
        
    )

    # 将推荐的范围保存到非结构化字典中备用
    hdata.uns["recommended_k"] = (min_k, max_k)
    return min_k, max_k


def init_model(hdata, n_metacells, **kwargs):
    """
    步骤 3: 初始化模型参数，实例化 MultiViewSEACells 并将其挂载到 hdata
    """
    # 保留您在 pipe.py 中的默认参数
    default_params = {}
    default_params.update(kwargs)

    # 实例化您原始的类，不修改任何内部结构
    hdata.model = MultiViewSEACells(n_metacells=n_metacells, **default_params)
    print(f"✅ 模型参数初始化完成，目标 MetaCell 数量: {n_metacells}")


def compute_kernels(hdata, use_views=None):
    """
    步骤 4: 计算核矩阵，将选定视图的 PCA 矩阵分别建 RBF 核传给模型。

    参数:
        use_views: 控制哪些视图参与核矩阵计算。
            None (默认) — 只使用 modality='hic' 的视图（原始行为）
            'all'       — hdata.views_pca 中所有视图都参与（含 RNA/METH/ATAC）
            list[str]   — 指定视图键列表，如 ['hic_50000', 'hic_500000', 'rna']

    示例:
        sk.tl.compute_kernels(hdata)                          # 仅 HiC
        sk.tl.compute_kernels(hdata, use_views='all')         # 所有视图
        sk.tl.compute_kernels(hdata, use_views=['hic_500000', 'rna'])  # 指定
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化，请先运行 sk.tl.init_model(hdata, ...)")

    if use_views is None:
        # 默认：只用 hic 模态
        selected_keys = [
            k for k in hdata.views_pca
            if hdata.view_configs.get(k, {}).get('modality', MODALITY_HIC) == MODALITY_HIC
        ]
        if not selected_keys:
            # 无配置信息时全部使用（兼容旧数据）
            selected_keys = list(hdata.views_pca.keys())
    elif use_views == 'all':
        selected_keys = list(hdata.views_pca.keys())
    else:
        # 用户显式指定，做存在性校验
        missing = [k for k in use_views if k not in hdata.views_pca]
        if missing:
            raise KeyError(f"以下视图在 hdata.views_pca 中不存在: {missing}\n"
                           f"现有视图: {list(hdata.views_pca.keys())}")
        selected_keys = list(use_views)

    pca_list = [normalize(hdata.views_pca[k], norm="l2", axis=1) for k in selected_keys]
    # 将实际使用的视图键记录到 uns，供后续查看
    hdata.uns['kernel_views'] = selected_keys
    print(f"  参与核矩阵计算的视图 ({len(selected_keys)} 个): {selected_keys}")

    # 完全调用底层的核矩阵计算
    hdata.model.compute_kernels(pca_list, save_dir=None)



def initialize_waypoints(
    hdata, data_type="pca", seed=32, n_micro_clusters=None, ref_view_res=500000,
    init_method='minibatch_fps'
):
    """
    步骤 5: 模型 initialize + 顺带调用其可视化确认 waypoint

    init_method: waypoint 初始化策略
        'minibatch_fps' (默认): MiniBatchKMeans 微簇 + FPS (本算法)
        'greedy_fps'          : 直接全局贪心 FPS (SEACells 风格)
        'random'              : 随机均匀采样 (基线)
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化")

    if n_micro_clusters is None:
        n_micro_clusters = hdata.model.n_metacells

    # 调用底层 initialize
    hdata.model.initialize(
        seed=seed, data_type=data_type, n_micro_clusters=n_micro_clusters,
        init_method=init_method
    )

    # 按照您的流程，这一步直接出图确认


def fit(hdata, n_threads=10):
    """
    步骤 6: 进行模型拟合，并将结果（metacell 分配标签）持久化存回 HData.obs
    同时初始化 Metacell 的基础统计属性。
    Metacell ID 从 1 开始编号。
    """
    if hdata.model is None:
        raise ValueError("模型尚未初始化，请先运行 sk.tl.init_model(hdata, ...)")

    hdata.model.fit(n_threads=n_threads)
    # ID 从 1 开始：原始 labels 是 0-indexed，整体 +1
    hdata.obs["metacell"] = hdata.model.labels + 1

    # 初始化 hdata.metacells 基础统计表
    meta_stats = hdata.obs.groupby("metacell").agg({"depth": ["sum", "count", "mean"]})
    meta_stats.columns = ["total_depth", "cell_count", "mean_depth"]

    if "label" in hdata.obs.columns:

        def get_dominant(x):
            return x.value_counts().index[0]

        meta_stats["dominant_label"] = hdata.obs.groupby("metacell")["label"].apply(
            get_dominant
        )

    hdata.metacells = meta_stats
    print(
        "✅ 模型拟合完成，Metacell 标签（1起编号）已保存，基础属性(深度、组成等)已初始化至 hdata.metacells。"
    )


def calculate_metrics(hdata, cell_types):
    """
    步骤1: 计算核心评估指标，并将其缓存为模型属性 (纯计算，不绘图)

    参数:
    - cell_types: array-like, 细胞的真实类型标签

    返回:
    - purity_df: 包含各项纯度和大小评估指标的 DataFrame
    """

    print("\n" + "=" * 60)
    print("正在计算评估指标...")
    print("=" * 60)
    labels = hdata.obs["metacell"].values
    # 整理基础 DataFrame
    df = pd.DataFrame({"CellType": cell_types, "Metacell": labels})

    overmerge = calculate_overmerging_metrics(df["Metacell"], df["CellType"])
    wcos = overmerge["WCOS"]
    hwis = overmerge["HWIS"]

    def celltype_frac(x):
        val_counts = x["CellType"].value_counts()
        return val_counts.values[0] / val_counts.values.sum()

    def dominant_celltype(x):
        return x["CellType"].value_counts().index[0]

    # 聚合计算
    celltype_fraction = df.groupby("Metacell").apply(celltype_frac)
    celltype_dom = df.groupby("Metacell").apply(dominant_celltype)
    cell_num = df.groupby("Metacell").count()["CellType"]

    purity = pd.concat([celltype_dom, celltype_fraction, cell_num], axis=1)
    purity.columns = ["CellType", "CellType_purity", "cell_num"]

    # 动态计算惩罚因子与基线调整
    avg_size = purity["cell_num"].mean()
    thre = 2 * avg_size

    # 1. 过小惩罚
    purity["w_min"] = 1 - (1 / np.sqrt(purity["cell_num"]))
    purity.loc[purity["cell_num"] == 1, "w_min"] = 0.0  # 处理特例

    # 2. 过大惩罚
    excess_ratio = (purity["cell_num"] - thre) / avg_size
    excess_ratio = excess_ratio.clip(lower=0)
    purity["w_max"] = 1 / (1 + excess_ratio)

    # 3. 机会校正基线
    num_unique_types = df["CellType"].nunique()
    baseline = 1.0 / num_unique_types
    purity["P_adj"] = (purity["CellType_purity"] - baseline) / (1 - baseline)
    purity["P_adj"] = purity["P_adj"].clip(lower=0)

    # 4. 最终核心 EP_v2
    purity["EP_v2"] = purity["P_adj"] * purity["w_min"] * purity["w_max"]

    # 记录内部属性
    mean_purity_ = purity["CellType_purity"].mean()
    global_score_ = (purity["EP_v2"] * purity["cell_num"]).sum() / purity[
        "cell_num"
    ].sum()

    # 计算 Accuracy 并映射标签以便画图
    hash_meta = purity["CellType"].to_dict()
    df["meta_lb"] = df["Metacell"].map(hash_meta)
    accuracy_ = (df["CellType"] == df["meta_lb"]).sum() / df.shape[0]

    print(f"✅ 指标计算完成！(发现 {num_unique_types} 种细胞类型)")
    return purity, df, avg_size, thre, wcos, hwis


def evaluate(hdata, true_labels):
    """
    步骤 9: 评估模型并计算纯度
    """

    # 确保结构指标 (compactness & separation) 已计算
    evaluate_metacell(hdata)

    purity_df, eval_df_cache, avg_size_cache, thre_cache, wcos, hwis = (
        calculate_metrics(hdata, true_labels)
    )
    accuracy = (
        eval_df_cache["CellType"] == eval_df_cache["meta_lb"]
    ).sum() / eval_df_cache.shape[0]
    global_score = (purity_df["EP_v2"] * purity_df["cell_num"]).sum() / purity_df[
        "cell_num"
    ].sum()
    comp = hdata.uns.get("mean_compactness", np.nan)
    sep = hdata.uns.get("mean_separation", np.nan)

    print("-" * 40)
    print(f"简单平均纯度 (Mean Purity)  : {purity_df['CellType_purity'].mean():.4f}")
    print(f"模型准确率 (Accuracy)      : {accuracy:.4f}")
    print(f"全局加权分 (Global Score)  : {global_score:.4f}")
    print(f"过度融合指标 (WCOS)       : {wcos:.4f}")
    print(f"Hub 权重不纯度 (HWIS)     : {hwis:.4f}")
    if not pd.isna(comp):
        print(f"紧凑度 (Compactness)      : {comp:.4f}")
    if not pd.isna(sep):
        print(f"分离度 (Separation)       : {sep:.4f}")
    print("-" * 40)

    metrics_summary = {
        "mean_purity": purity_df["CellType_purity"].mean(),
        "accuracy": accuracy,
        "global_score": global_score,
        "WCOS": wcos,
        "HWIS": hwis,
        "compactness": comp,
        "separation": sep,
    }

    hdata.uns["purity_df"] = purity_df
    if "metrics" not in hdata.uns:
        hdata.uns["metrics"] = {}
    hdata.uns["metrics"].update(metrics_summary)
    hdata.uns["eval_df_cache"] = eval_df_cache
    hdata.uns["avg_size_cache"] = avg_size_cache
    hdata.uns["thre_cache"] = thre_cache
    hdata.uns["accuracy"] = accuracy
    hdata.uns["global_score"] = global_score
    hdata.uns["wcos"] = wcos
    hdata.uns["hwis"] = hwis

    # ==============================================================
    # 新增：将 purity 核心指标无缝追加到现有的 hdata.metacells 中
    # ==============================================================
    # 过滤掉与基础属性重复的列 (cell_num 对应 cell_count, CellType 对应 dominant_label)
    cols_to_add = [c for c in purity_df.columns if c not in ["CellType", "cell_num"]]

    if "hdata.metacells" not in hdata.__dict__ or hdata.metacells.empty:
        hdata.metacells = purity_df.copy()

    if hdata.metacells.empty:
        hdata.metacells = purity_df.copy()  # 防御性编程：如果用户没正常走 fit
    else:
        for col in cols_to_add:
            hdata.metacells[col] = purity_df[col]

    print("✅ 评估指标计算完成，纯度得分(EP_v2等)已同步至 hdata.metacells。")
    return purity_df, metrics_summary
