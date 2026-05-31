import math
import cooler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.colors import LogNorm
from ..utils.tad import compute_insulation_score
from ..core.hdata import _make_view_key, MODALITY_HIC


def _resolve_view_key(hdata, resolution, view_dict_name='views_umap'):
    """
    将 int 分辨率（旧接口）转换为 views_* 字典里实际存在的键。
    - int 500000  -> 'hic_500000'（若存在），否则回退到直接 int
    - str 'rna'   -> 直通
    """
    view_dict = getattr(hdata, view_dict_name, {})
    if isinstance(resolution, int):
        str_key = _make_view_key(MODALITY_HIC, resolution)
        if str_key in view_dict:
            return str_key
        if resolution in view_dict:   # 向后兼容
            return resolution
        raise KeyError(f"{view_dict_name} 中未找到视图 '{str_key}'，"
                       f"现有键: {list(view_dict.keys())}")
    if resolution in view_dict:
        return resolution
    raise KeyError(f"{view_dict_name} 中未找到视图 '{resolution}'，"
                   f"现有键: {list(view_dict.keys())}")


def _fetch_metacell_region_matrix(hdata, metacell_id, chrom, start, end, resolution,
                                  base_on='pair', balance=True):
    """读取指定 metacell 的局部接触矩阵，失败时返回 None。"""
    if base_on == 'pair':
        if metacell_id not in hdata.metacell_data.get('mcool', {}):
            return None
        mcool_path = hdata.metacell_data['mcool'][metacell_id]
        uri = f"{mcool_path}::/resolutions/{resolution}"
        clr = cooler.Cooler(uri)
        return clr.matrix(balance=balance).fetch((chrom, start, end))

    if base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
        str_res = str(resolution)
        if base_on not in hdata.metacell_data or str_res not in hdata.metacell_data[base_on]:
            return None

        mcool_dict = hdata.metacell_data[base_on][str_res]
        if metacell_id not in mcool_dict or chrom not in mcool_dict[metacell_id]:
            return None

        whole_chrom_mat = mcool_dict[metacell_id][chrom]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        max_bins = whole_chrom_mat.shape[0]
        start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)

        import scipy.sparse as sp
        if sp.issparse(whole_chrom_mat):
            return whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
        return whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]

    raise ValueError("base_on 必须是 'pair'、'mat'、'mat_redist'、'mat_consensus' 或 'mat_EM'")


def _plot_upper_triangle_rot45(ax, mat, cmap='Reds', vmin=None, vmax=None):
    """绘制上三角 45 度旋转热图（主对角线与水平线平行）。"""
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("输入矩阵必须是方阵。")

    # 仅保留上三角（含对角线）
    upper = mat.copy()
    upper[np.tril_indices(n, k=-1)] = np.nan

    # 将(i, j)映射到旋转坐标: x=(i+j)/2, y=(j-i)/2
    i_edge, j_edge = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    x_edge = (i_edge + j_edge) / 2.0
    y_edge = (j_edge - i_edge) / 2.0

    mappable = ax.pcolormesh(x_edge, y_edge, upper, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, n / 2.0)
    ax.set_aspect('equal')
    ax.margins(x=0, y=0)
    ax.set_xticks([])
    ax.set_yticks([])
    return mappable


def _fill_nan_nearest_1d(arr):
    """使用最近邻填充一维数组中的 NaN，保证色带不出现断裂空白。"""
    out = np.asarray(arr, dtype=float).copy()
    n = out.shape[0]
    if n == 0:
        return out

    valid = ~np.isnan(out)
    if not np.any(valid):
        return np.zeros_like(out)

    first_valid = np.where(valid)[0][0]
    last_valid = np.where(valid)[0][-1]
    out[:first_valid] = out[first_valid]
    out[last_valid + 1:] = out[last_valid]

    for i in range(first_valid + 1, last_valid + 1):
        if np.isnan(out[i]):
            out[i] = out[i - 1]

    for i in range(last_valid - 1, first_valid - 1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]

    return out


def plot_celltype_triangle_heatmaps(
    hdata,
    cell_type,
    chrom,
    start,
    end,
    resolution,
    balance=True,
    base_on='pair',
    ncols=4,
    cell_type_col='cell_type',
    cmap='Reds',
    log1p=True,
    fill_diagonal_zero=True,
    vmax=None,
    vmin=None,
    ins_window=10,
    ins_normalize=True,
    ins_flank_bins=0,
    ins_cmap='coolwarm',
    ins_vmin=None,
    ins_vmax=None,
    strip_height=0.22,
    strip_gap=0.0,
    save_path=None,
    dpi=300,
):
    """
        绘制指定细胞类型下全部 Metacell 的“上三角旋转 45 度”热图，并在下方叠加 insulation score 色块条带。

    每个 metacell 子图由两部分组成:
    1) 上部: 仅显示上三角接触矩阵，旋转后主对角线为水平线
        2) 下部: 相同区域的 insulation score 色块（每个 bin 一格）

        说明:
        - 为避免边界效应导致长度错位，可设置 ins_flank_bins > 0，
            即先在 [start-flank, end+flank] 上计算 IS，再裁剪回 [start, end]。
        - strip_height 控制 IS 条带厚度（相对比例）。
        - strip_gap 控制三角热图与 IS 条带的间距（相对三角图高度）。
            正值更远，0 为紧贴，负值更近（可轻微重叠）。
    """
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")

    target_obs = hdata.metacells[hdata.metacells[cell_type_col] == cell_type]
    m_ids = target_obs.index.tolist()
    if not m_ids:
        print(f"未找到细胞类型为 '{cell_type}' 的 Metacell，请检查名称是否正确。")
        return

    print(f"共找到 {len(m_ids)} 个属于 '{cell_type}' 的 Metacells, 准备渲染三角热图与 insulation 色块...")

    nrows = math.ceil(len(m_ids) / ncols)
    fig = plt.figure(figsize=(ncols * 4.4, nrows * 4.4))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.25, hspace=0.35)

    last_mappable = None
    last_ins_mappable = None

    for i, m_id in enumerate(m_ids):
        r, c = divmod(i, ncols)
        # 使用 height_ratios 和 hspace 精确控制子图大小和间距
        hm_ratio = 4.0
        strip_ratio = max(0.02, float(strip_height))
        hm_to_strip_gap = float(strip_gap) * hm_ratio  # 转换间距为比例
        inner = outer[r, c].subgridspec(
            2, 1,
            height_ratios=[hm_ratio, strip_ratio],
            hspace=hm_to_strip_gap
        )
        ax_hm = fig.add_subplot(inner[0])
        ax_ins = fig.add_subplot(inner[1], sharex=ax_hm)

        try:
            mat = _fetch_metacell_region_matrix(
                hdata,
                metacell_id=m_id,
                chrom=chrom,
                start=start,
                end=end,
                resolution=resolution,
                base_on=base_on,
                balance=balance,
            )
            if mat is None:
                ax_hm.set_title(f"{m_id} (No data)")
                ax_hm.axis('off')
                ax_ins.axis('off')
                continue

            mat = np.nan_to_num(np.asarray(mat, dtype=float))
            if log1p:
                mat = np.log1p(mat)
            if fill_diagonal_zero:
                np.fill_diagonal(mat, 0)

            last_mappable = _plot_upper_triangle_rot45(
                ax_hm,
                mat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax_hm.set_title(str(m_id), fontsize=10)

            n_bins = mat.shape[0]

            # 可选：使用扩展区间计算 IS，再裁剪到目标区间，保证与主图长度对齐
            # 至少扩展到 ins_window，减少边界 NaN。
            flank_bins_effective = max(int(ins_flank_bins), int(ins_window))
            if flank_bins_effective > 0:
                ext_start = max(0, int(start - flank_bins_effective * resolution))
                ext_end = int(end + flank_bins_effective * resolution)
                ext_mat = _fetch_metacell_region_matrix(
                    hdata,
                    metacell_id=m_id,
                    chrom=chrom,
                    start=ext_start,
                    end=ext_end,
                    resolution=resolution,
                    base_on=base_on,
                    balance=balance,
                )

                if ext_mat is not None:
                    ext_mat = np.nan_to_num(np.asarray(ext_mat, dtype=float))
                    if log1p:
                        ext_mat = np.log1p(ext_mat)
                    if fill_diagonal_zero:
                        np.fill_diagonal(ext_mat, 0)

                    ins_ext = compute_insulation_score(ext_mat, window=ins_window, normalize=ins_normalize)
                    start_offset = int((start - ext_start) // resolution)
                    ins = ins_ext[start_offset:start_offset + n_bins]
                else:
                    ins = compute_insulation_score(mat, window=ins_window, normalize=ins_normalize)
            else:
                ins = compute_insulation_score(mat, window=ins_window, normalize=ins_normalize)

            # 兜底: 保证 IS 长度与热图对角线长度一致
            if len(ins) < n_bins:
                pad = np.full(n_bins - len(ins), np.nan)
                ins = np.concatenate([ins, pad])
            elif len(ins) > n_bins:
                ins = ins[:n_bins]

            # 将边界 NaN 连续填充，避免色块看起来“变短”
            ins = _fill_nan_nearest_1d(ins)

            # 下方使用“每 bin 一格”的色块展示 IS
            # 使用 pcolormesh 按 bin 边界渲染，确保与上图 x 轴严格一致
            x_edges = np.arange(n_bins + 1)
            y_edges = np.array([0.0, 1.0])
            ins_strip = ins[np.newaxis, :]
            last_ins_mappable = ax_ins.pcolormesh(
                x_edges,
                y_edges,
                ins_strip,
                shading='flat',
                cmap=ins_cmap,
                vmin=ins_vmin,
                vmax=ins_vmax,
                antialiased=False,
            )
            ax_ins.set_xlim(0, n_bins)
            ax_ins.set_ylim(0, 1)
            ax_ins.margins(x=0, y=0)
            ax_ins.tick_params(axis='x', labelbottom=False)
            ax_ins.set_yticks([])
            ax_ins.spines['top'].set_visible(False)
            ax_ins.spines['right'].set_visible(False)
            ax_ins.spines['left'].set_visible(False)
            ax_ins.set_ylabel('IS', fontsize=8, rotation=0, labelpad=8)

        except Exception:
            ax_hm.set_title(f"{m_id} (Error)")
            ax_hm.axis('off')
            ax_ins.axis('off')

    # 隐藏多余网格
    for j in range(len(m_ids), nrows * ncols):
        rr, cc = divmod(j, ncols)
        ax_empty = fig.add_subplot(outer[rr, cc])
        ax_empty.axis('off')

    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, ax=fig.axes, shrink=0.45, pad=0.01)
        cbar.ax.tick_params(labelsize=8)

    if last_ins_mappable is not None:
        # 给 IS 条带单独加一个横向 colorbar，避免和主热图 colorbar 混淆。
        cax_ins = fig.add_axes([0.12, 0.03, 0.35, 0.015])
        cbar_ins = fig.colorbar(last_ins_mappable, cax=cax_ins, orientation='horizontal')
        cbar_ins.ax.tick_params(labelsize=8)
        cbar_ins.set_label('IS', fontsize=8)

    fig.suptitle(
        f"Triangle Heatmap + Insulation | Cell Type: {cell_type} | {chrom}:{start}-{end} @ {resolution//1000}kb",
        y=1.01,
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_metacell_insulation_strips(
    hdata,
    chrom,
    start,
    end,
    resolution,
    balance=True,
    base_on='pair',
    cell_type_col='cell_type',
    metacell_order=None,
    ins_window=10,
    ins_normalize=True,
    ins_flank_bins=0,
    log1p=True,
    fill_diagonal_zero=True,
    ins_cmap='coolwarm',
    ins_vmin=None,
    ins_vmax=None,
    row_height=0.3,
    label_fontsize=9,
    show_colorbar=True,
    figsize=None,
    save_path=None,
    dpi=300,
):
    """
    将多个 Metacell 的 insulation score 堆叠为一张 heatmap 展示。

    所有行组成一个矩阵，用单次 pcolormesh 渲染，无行间间隔。
    Y 轴标签为细胞类型，同类型的行共用一个居中标签。

    Parameters
    ----------
    hdata : HiCData
        含有 metacells 属性的数据对象。
    chrom, start, end, resolution : 基因组区间参数。
    balance : bool
        是否使用 balanced 矩阵。
    base_on : str
        矩阵来源。
    cell_type_col : str
        hdata.metacells 中存储细胞类型的列名。
    metacell_order : list or None
        指定 metacell id 的顺序列表（从上到下）。
        若为 None，自动按 cell_type_col 分组排序。
    ins_window : int
        insulation score 计算窗口大小（bin 数）。
    ins_normalize : bool
        是否对 insulation score 归一化。
    ins_flank_bins : int
        额外扩展 bins 数，用于减少边界 NaN（至少取 ins_window）。
    log1p : bool
        读取矩阵时是否 log1p 变换。
    fill_diagonal_zero : bool
        是否将对角线置零。
    ins_cmap : str
        IS 色块 colormap。
    ins_vmin, ins_vmax : float or None
        IS colormap 的上下界，None 则自动从数据范围推断。
    row_height : float
        每行的高度（英寸），默认 0.3。
    label_fontsize : int
        Y 轴细胞类型标签的字体大小。
    show_colorbar : bool
        是否显示 colorbar。
    figsize : tuple or None
        图像尺寸，None 则自动推断。
    save_path : str or None
        保存路径。
    dpi : int
        保存分辨率。
    """
    if not hasattr(hdata, 'metacells'):
        raise ValueError("hdata 没有 metacells 属性，请检查数据。")

    # 确定 metacell 顺序
    if metacell_order is not None:
        m_ids = list(metacell_order)
    else:
        if cell_type_col in hdata.metacells.columns:
            m_ids = hdata.metacells.sort_values(cell_type_col).index.tolist()
        else:
            m_ids = hdata.metacells.index.tolist()

    n_strips = len(m_ids)
    if n_strips == 0:
        print("没有可用的 Metacell。")
        return

    # id -> 细胞类型映射
    id_to_ct = {}
    if cell_type_col in hdata.metacells.columns:
        id_to_ct = hdata.metacells[cell_type_col].to_dict()

    # 基因组参数
    n_bins = int((end - start) // resolution)
    flank_bins_effective = max(int(ins_flank_bins), int(ins_window))
    ext_start = max(0, int(start - flank_bins_effective * resolution))
    ext_end = int(end + flank_bins_effective * resolution)
    start_offset = int((start - ext_start) // resolution)

    # 收集所有 IS，组成矩阵 (n_strips x n_bins)
    all_ins = []
    for m_id in m_ids:
        try:
            ext_mat = _fetch_metacell_region_matrix(
                hdata, metacell_id=m_id, chrom=chrom,
                start=ext_start, end=ext_end, resolution=resolution,
                base_on=base_on, balance=balance,
            )
            if ext_mat is None:
                all_ins.append(np.full(n_bins, np.nan))
                continue
            ext_mat = np.nan_to_num(np.asarray(ext_mat, dtype=float))
            if log1p:
                ext_mat = np.log1p(ext_mat)
            if fill_diagonal_zero:
                np.fill_diagonal(ext_mat, 0)
            ins_ext = compute_insulation_score(ext_mat, window=ins_window, normalize=ins_normalize)
            ins = ins_ext[start_offset:start_offset + n_bins]
            if len(ins) < n_bins:
                ins = np.concatenate([ins, np.full(n_bins - len(ins), np.nan)])
            elif len(ins) > n_bins:
                ins = ins[:n_bins]
            ins = _fill_nan_nearest_1d(ins)
        except Exception:
            ins = np.full(n_bins, np.nan)
        all_ins.append(ins)

    mat2d = np.vstack(all_ins)  # shape: (n_strips, n_bins)

    # 全局 vmin/vmax
    if ins_vmin is None or ins_vmax is None:
        flat = mat2d[~np.isnan(mat2d)]
        if ins_vmin is None:
            ins_vmin = float(np.nanmin(flat)) if len(flat) else -1
        if ins_vmax is None:
            ins_vmax = float(np.nanmax(flat)) if len(flat) else 1

    # 图像尺寸
    if figsize is None:
        fig_w = max(8, min(24, n_bins * 0.08))
        fig_h = max(2, n_strips * row_height + 1.0)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    x_edges = np.arange(n_bins + 1)
    y_edges = np.arange(n_strips + 1)  # 0..n_strips，每行占 1 个单位

    mappable = ax.pcolormesh(
        x_edges, y_edges, mat2d,
        shading='flat',
        cmap=ins_cmap,
        vmin=ins_vmin,
        vmax=ins_vmax,
        antialiased=False,
    )
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_strips)
    ax.margins(x=0, y=0)
    ax.invert_yaxis()  # 第一行在上方

    # Y 轴：按细胞类型分组，每组居中显示一个标签
    ct_list = [id_to_ct.get(m, '') for m in m_ids]
    # 找出每个细胞类型组的起止行，计算居中位置
    group_ticks = []   # tick 位置（行中心）
    group_labels = []  # 标签
    i = 0
    while i < n_strips:
        ct = ct_list[i]
        j = i + 1
        while j < n_strips and ct_list[j] == ct:
            j += 1
        center = (i + j) / 2.0  # 中心（已 invert，坐标仍为正向 0..n_strips）
        group_ticks.append(center)
        group_labels.append(ct if ct else str(m_ids[i]))
        i = j

    ax.set_yticks(group_ticks)
    ax.set_yticklabels(group_labels, fontsize=label_fontsize)
    ax.tick_params(axis='y', length=0)  # 不显示刻度线

    # 在细胞类型边界画分隔线
    boundaries = set()
    for k in range(1, n_strips):
        if ct_list[k] != ct_list[k - 1]:
            boundaries.add(k)
    for b in boundaries:
        ax.axhline(b, color='white', linewidth=1.5, lw=1.5)

    # X 轴坐标
    tick_positions = np.linspace(0, n_bins, min(10, n_bins + 1), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{(start + t * resolution) // 1000}kb" for t in tick_positions],
        rotation=30, ha='right', fontsize=8,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('IS', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Insulation Score Heatmap | {chrom}:{start}-{end} @ {resolution // 1000}kb",
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_basic_purity(hdata,figsize=(8, 6), save_path=None, dpi=300):
    """可视化 1: 基础纯度柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='CellType_purity', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('Basic Purity by Cell Type')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_views(hdata, label=None, ncols=3, save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()



def plot_depth_distribution(hdata, save_path=None, dpi=300):
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# ==========================================================
# 2. 模型核心结果可视化 (代理底层 plotting.py 中的方法)
# ==========================================================



def plot_ep_score(hdata,figsize=(12, 6), save_path=None, dpi=300):
    """可视化 3: 最终评估得分 EP_v2 柱状图"""
    purity_df = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='CellType', y='EP_v2', data=purity_df)
    plt.xticks(rotation=45)
    plt.title('EP v2 Score by Cell Type (Corrected & Penalized)')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_umap_assignment(hdata,resolution=None,  figsize=(7,7), save_path=None, dpi=300):
    """
    可视化 4: 单点 UMAP 散点图 (按模型分配的 Metacell ID 染色)
    注: 该图只依赖模型结果，可在 calculate_metrics 之前调用
    """
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    
    labels = hdata.obs['label'].values
    
    umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    sns.scatterplot(x=umap_coords[:, 0], y=umap_coords[:, 1], 
                    hue=labels, palette='tab20', s=5, legend=False, rasterized=True)
    plt.title("UMAP: Metacell Assignments")
    plt.axis('off')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()





def plot_umap_comparison(hdata, resolution=None,  figsize=(14, 6), save_path=None, dpi=300):

        """
        可视化 5: 双点 UMAP 对比图 (左侧为预测平滑的类型，右侧为真实单细胞类型)
        """
        if resolution is None:
            raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")

        umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        df = hdata.uns['eval_df_cache']
        
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
        if 'save_path' in locals() and save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
 





def plot_metacell_sizes(hdata, figsize=(8, 6), bins=20, save_path=None, dpi=300):
    purity_df_ = hdata.uns['purity_df']
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    plt.hist(purity_df_['cell_num'], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of Metacell Sizes')
    plt.xlabel('Number of Cells')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
        
        



def plot_initialization(hdata, resolution=None, title="Initialization Waypoints",figsize=(8, 6), save_path=None, dpi=300):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
    
    waypoints = hdata.model.waypoints
    
    plt.figure(figsize=figsize)
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
    wp_coords = umap_coords[waypoints, :]
    plt.scatter(wp_coords[:, 0], wp_coords[:, 1], c='red', s=60, edgecolors='black', linewidth=1, label='Initial Waypoints', zorder=10)
    plt.title(title)
    plt.axis('off')
    plt.legend()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_specific_metacell(hdata, metacell_id,resolution=None, figsize=(8, 6), save_path=None, dpi=300):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell'].values
    umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
    indices = np.where(labels == metacell_id)[0]
    
    plt.figure(figsize=figsize)
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.3, label='Background')
    
    if len(indices) > 0:
        target_coords = umap_coords[indices]
        plt.scatter(target_coords[:, 0], target_coords[:, 1], c='red', s=20, label=f'Metacell {metacell_id}')
        center = np.mean(target_coords, axis=0)
        plt.scatter(center[0], center[1], c='black', marker='x', s=100, linewidth=2, label='Centroid')
        
    plt.title(f"Visual Diagnosis: Metacell {metacell_id}")
    plt.legend()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()






def plot_metacells(hdata,resolution=None, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False, save_path=None, dpi=300):
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell']
    umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
    metacell_coords = []
    metacell_counts = []
    present_indices = np.unique(labels)
    
    for k in present_indices:
        indices = np.where(labels == k)[0]
        metacell_coords.append(np.mean(umap_coords[indices], axis=0))
        metacell_counts.append(len(indices))
    
    metacell_coords = np.array(metacell_coords)
    metacell_counts = np.array(metacell_counts)
    
    if len(metacell_counts) == 0:
        print("警告: 没有发现活跃的 metacells。")
        return

    if len(metacell_counts) > 1 and metacell_counts.max() > metacell_counts.min():
        norm_sizes = (metacell_counts - metacell_counts.min()) / (metacell_counts.max() - metacell_counts.min())
        plot_sizes = min_size + norm_sizes * (max_size - min_size)
    else:
        plot_sizes = np.full(len(metacell_counts), (min_size + max_size) / 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
    plt.scatter(metacell_coords[:, 0], metacell_coords[:, 1], 
                c='blue', s=plot_sizes, edgecolors='white', linewidth=1, alpha=0.8, zorder=10)
    
    min_c = metacell_counts.min()
    max_c = metacell_counts.max()
    mid_c = int((min_c + max_c) / 2)
    legend_sizes = [min_size, (min_size+max_size)/2, max_size]
    legend_labels = [f'{min_c} cells', f'{mid_c} cells', f'{max_c} cells']
    
    handles = []
    for s, l in zip(legend_sizes, legend_labels):
        handles.append(plt.scatter([], [], c='blue', alpha=0.8, s=s, edgecolors='white', label=l))
    handles.append(plt.scatter([], [], c='lightgrey', s=20, label='Single Cells'))
    
    plt.legend(handles=handles, title="Metacell Size (Count)", loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.5, borderpad=1)
    
    if show_idx:
        for i, k in enumerate(present_indices):
            plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(k), 
                        fontsize=10, ha='center', va='center', color='black', fontweight='bold', zorder=20)
    
    plt.title(f"{title}\n(Metacells: {len(metacell_coords)}, Count Range: {min_c}-{max_c})")
    plt.axis('off')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_metacells2(hdata, resolution=None, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False, show_count=False, label_col='label', cell_alpha=0.3, metacell_alpha=0.8, save_path=None, dpi=300):
    """
    可视化 Metacell 在 UMAP 上的位置，并用主导细胞类型 (dominant label) 染色，
    同时将底层的单细胞也一并着色展示。
    """
    if resolution is None:
        raise ValueError("请提供 resolution 参数以获取对应的 UMAP 坐标")
    labels = hdata.obs['metacell']
    umap_coords = hdata.views_umap[_resolve_view_key(hdata, resolution, 'views_umap')]
    metacell_coords = []
    metacell_counts = []
    metacell_dominant_labels = []
    present_indices = np.unique(labels)
    
    import pandas as pd
    has_labels = False
    if label_col in hdata.obs.columns:
        cell_labels = hdata.obs[label_col].values
        has_labels = True
        
    for k in present_indices:
        indices = np.where(labels == k)[0]
        metacell_coords.append(np.mean(umap_coords[indices], axis=0))
        metacell_counts.append(len(indices))
        if has_labels:
            dom_label = pd.Series(cell_labels[indices]).value_counts().index[0]
            metacell_dominant_labels.append(dom_label)
        else:
            metacell_dominant_labels.append("Unknown")
            
    metacell_coords = np.array(metacell_coords)
    metacell_counts = np.array(metacell_counts)
    
    if len(metacell_counts) == 0:
        print("警告: 没有发现活跃的 metacells。")
        return

    if len(metacell_counts) > 1 and metacell_counts.max() > metacell_counts.min():
        norm_sizes = (metacell_counts - metacell_counts.min()) / (metacell_counts.max() - metacell_counts.min())
        plot_sizes = min_size + norm_sizes * (max_size - min_size)
    else:
        plot_sizes = np.full(len(metacell_counts), (min_size + max_size) / 2)

    plt.figure(figsize=(10, 8))
    
    # 提前处理颜色映射
    mc_colors = ['blue'] * len(metacell_coords)
    cell_colors_list = 'lightgrey'
    handles = []
    
    if has_labels:
        if 'eval_df_cache' in hdata.uns:
            unified_hue_order = sorted(hdata.uns['eval_df_cache']['CellType'].dropna().unique())
        else:
            unified_hue_order = sorted(pd.Series(cell_labels).dropna().unique())
            
        palette = sns.color_palette('tab20', n_colors=len(unified_hue_order))
        color_map = {lbl: palette[i] for i, lbl in enumerate(unified_hue_order)}
        
        mc_colors = [color_map.get(lbl, 'blue') for lbl in metacell_dominant_labels]
        cell_colors_list = [color_map.get(lbl, 'lightgrey') for lbl in cell_labels]
        
        # 添加类别图例
        for lbl in unified_hue_order:
            handles.append(plt.scatter([], [], c=[color_map[lbl]], s=100, edgecolors='none', linewidth=0, label=lbl))
        # 空白分隔
        handles.append(plt.scatter([], [], c='white', s=0, label=''))

    # 绘制底层的单细胞
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=cell_colors_list, s=5, alpha=cell_alpha, rasterized=True, edgecolors='none', linewidth=0)

    # 绘制顶层的 Metacell
    plt.scatter(metacell_coords[:, 0], metacell_coords[:, 1],
            c=mc_colors, s=plot_sizes, edgecolors='none', linewidths=0, alpha=metacell_alpha, zorder=10)
    min_c = metacell_counts.min()
    max_c = metacell_counts.max()
    mid_c = int((min_c + max_c) / 2)
    legend_sizes = [min_size, (min_size+max_size)/2, max_size]
    legend_labels = [f'{min_c} cells', f'{mid_c} cells', f'{max_c} cells']
    
    # 尺寸图例
    for s, l in zip(legend_sizes, legend_labels):
        handles.append(plt.scatter([], [], c='gray', alpha=0.8, s=s, edgecolors='none', linewidth=0, label=l))
    handles.append(plt.scatter([], [], c='lightgrey', s=20, edgecolors='none', linewidth=0, label='Single Cells'))
    
    plt.legend(handles=handles, title="Metacell Legend", loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.0, borderpad=1)
    
    if show_idx:
        for i, k in enumerate(present_indices):
            plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(k), 
                        fontsize=10, ha='center', va='center', color='black', fontweight='bold', zorder=20)
    elif show_count:
        for i, cnt in enumerate(metacell_counts):
            plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(cnt),
                        fontsize=8, ha='center', va='center', color='black', fontweight='bold', zorder=20)
    
    plt.title(f"{title}\n(Metacells: {len(metacell_coords)}, Count Range: {min_c}-{max_c})")
    plt.axis('off')
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_metacell_heatmap(hdata, metacell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='Reds', vmin=None, vmax=None,log1p=True,fill_diagonal_zero=True,save_path=None, dpi=300, **kwargs):
    """
    可视化单个 Metacell 在指定区间的高分辨率 Hi-C 热图。
    
    参数:
    - hdata: 核心数据对象
    - metacell_id: 目标 metacell 的 ID (字符串或数字，会被强转对齐)
    - chrom: 染色体名称 (如 'chr1')
    - start: 起始位置 (bp)
    - end: 终止位置 (bp)
    - resolution: 分辨率
    - base_on: 'pair' (通过 mcool 读取) 或 'mat' (直接读取 views_mat 聚合结果)
    - balance: 是否使用 balance 后的矩阵 (布尔值，仅对 'pair' 模式生效)
    """
    if base_on == 'pair':
        if 'mcool' not in hdata.metacell_data or metacell_id not in hdata.metacell_data['mcool']:
            raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mcool 记录。请确保已经运行过 aggregate_metacell_pairs。")
            
        mcool_path = hdata.metacell_data['mcool'][metacell_id]
        try:
            uri = f"{mcool_path}::/resolutions/{resolution}"
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在 mcool 文件中读取分辨率 {resolution}。确保 pair 流程初始化了该分辨率。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
        str_res = str(resolution)
        dict_key = base_on
        if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
            raise ValueError(f"未找到基于 {base_on} 聚合的分辨率 {resolution} 数据。请先运行对应的聚合函数。")
            
        mcool_dict = hdata.metacell_data[dict_key][str_res]
        if metacell_id not in mcool_dict:
             raise ValueError(f"未找到 Metacell '{metacell_id}' 的 {base_on} 聚合记录。")
             
        if chrom not in mcool_dict[metacell_id]:
             raise ValueError(f"未找到染色体 {chrom} 的聚合矩阵。")
             
        whole_chrom_mat = mcool_dict[metacell_id][chrom]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) # 将 NaN 替换为 0
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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return mat

def plot_cell_of_metacell_heatmap(hdata, metacell_id, cell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='Reds', vmin=None, vmax=None, log1p=True, fill_diagonal_zero=True, save_path=None, dpi=300, **kwargs):
    """
    可视化指定 Metacell 内单个原始细胞的高分辨率 Hi-C 热图。
    
    参数:
    - hdata: 核心数据对象
    - metacell_id: 目标 metacell 的 ID (如果在 hdata.obs 中有记录可供校验，否则仅用于标题展示)
    - cell_id: 目标单细胞的 ID (在 hdata.obs.index 中的索引或名称)
    - 其他参数与 plot_metacell_heatmap 保持一致
    """
    mat = None
    if base_on == 'pair':
        # 尝试查找已生成的单细胞 cool/mcool 文件
        # 如果 cell_id 是整数索引，我们根据 hdata.data_dir 中的文件顺序推推断其对应的文件名
        cell_name = str(cell_id)
        if isinstance(cell_id, int) or str(cell_id).isdigit():
            idx = int(cell_id)
            all_files = []
            import os
            for val in sorted(os.listdir(hdata.data_dir)):
                if val.endswith('.pairs') or val.endswith('.pairs.gz'):
                    all_files.append(val)
            if idx < len(all_files):
                cell_name = all_files[idx].split('.pairs')[0]
        
        # 在相关目录中寻找匹配的 mcool/cool 文件
        mcool_path = None
        import os
        search_dirs = [hdata.data_dir, hdata.output_dir]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if f.endswith('.mcool') or f.endswith('.cool'):
                        if cell_name in f or str(cell_id) in f:
                            mcool_path = os.path.join(root, f)
                            break
                if mcool_path: break
            if mcool_path: break
            
        if not mcool_path:
            raise ValueError(f"未找到单细胞 '{cell_id}' 对应的 .mcool 或 .cool 文件，请确认已生成该文件或使用 base_on='mat'。")
            
        try:
            import cooler
            if mcool_path.endswith('.mcool'):
                uri = f"{mcool_path}::/resolutions/{resolution}"
            else:
                uri = f"{mcool_path}::/resolutions/{resolution}"
                if not cooler.fileops.is_multires_file(mcool_path):
                    uri = mcool_path
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在文件 {mcool_path} 中读取数据。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus']:
        if not hdata.views_mat:
            raise ValueError("hdata.views_mat 为空，请确认已运行预处理。")
        _vk = _resolve_view_key(hdata, resolution, 'views_mat')
        if chrom not in hdata.views_mat[_vk]:
            raise ValueError(f"未找到染色体 {chrom} 的 views_mat 数据。")
            
        try:
            cell_idx = hdata.obs.index.get_loc(cell_id)
        except KeyError:
            if isinstance(cell_id, int) and cell_id < len(hdata.obs):
                cell_idx = cell_id
            else:
                raise ValueError(f"Cell ID '{cell_id}' 不在 hdata.obs.index 中。")
                
        whole_chrom_mat = hdata.views_mat[_vk][chrom][cell_idx]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        # hdata.views_mat 中通常是稀疏矩阵，需转为密集矩阵并切片
        import scipy.sparse as sp
        if sp.issparse(whole_chrom_mat):
            mat = whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
        else:
            mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat)
    if log1p:
        mat = np.log1p(mat)
    if fill_diagonal_zero:
        np.fill_diagonal(mat, 0)
        
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    if vmax is None:
        if vmin is None:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax)
        else:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin)
    else:
        if vmin is None:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmax=vmax)
        else:
            sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax)
    
    ax.set_title(f"Metacell: {metacell_id} | Cell: {cell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return mat


def plot_celltype_heatmaps(hdata, cell_type, chrom, start, end, resolution, 
                            balance=True, base_on='pair', ncols=4, 
                            cell_type_col='cell_type', cmap='Reds',
                            log1p=True, fill_diagonal_zero=True,
                             vmax=None, vmin=None, save_path=None, dpi=300):
    """
    可视化指定细胞类型下所有 Metacell 的 Hi-C 热图 (以网格形式展示)。
    
    参数:
    - hdata: 核心数据对象
    - cell_type: 目标细胞类型名称
    - chrom: 染色体名称
    - start: 起始位置
    - end: 终止位置
    - resolution: 分辨率
    - base_on: 'pair' (通过 mcool 读取) 或 'mat' (直接读取 views_mat 聚合结果)
    - balance: 是否使用 balance 后的矩阵
    - ncols: 网格每一行展示的图表数量
    - cell_type_col: hdata.metacell_obs 中记录细胞类型的列名，默认为 'cell_type'
    """
    # 1. 获取对应细胞类型的所有 Metacell IDs
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
        
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
        
        try:
            if base_on == 'pair':
                if m_id not in hdata.metacell_data.get('mcool', {}):
                    ax.set_title(f"{m_id}\n(No mcool data)")
                    ax.axis('off')
                    continue
                mcool_path = hdata.metacell_data['mcool'][m_id]
                uri = f"{mcool_path}::/resolutions/{resolution}"
                clr = cooler.Cooler(uri)
                mat = clr.matrix(balance=balance).fetch((chrom, start, end))
                
            elif base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
                str_res = str(resolution)
                dict_key = base_on
                if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
                    raise ValueError(f"未找到基于 {base_on} 聚合的分辨率数据。")
                mcool_dict = hdata.metacell_data[dict_key][str_res]
                if m_id not in mcool_dict or chrom not in mcool_dict[m_id]:
                    ax.set_title(f"{m_id}\n(No mat data)")
                    ax.axis('off')
                    continue
                
                whole_chrom_mat = mcool_dict[m_id][chrom]
                start_bin = int(start // resolution)
                end_bin = int(np.ceil(end / resolution))
                max_bins = whole_chrom_mat.shape[0]
                start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)
                mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
            else:
                raise ValueError("base_on 必须是 'pair' 或 'mat'")

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
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_metacell_region(hdata, chrom, start, end, resolution, celltype=None, 
                         balance=True, base_on='pair', ncols=4, named_on=None,
                         cell_type_col='cell_type', cmap='Reds',
                         log1p=True, fill_diagonal_zero=True,
                         vmax=None, vmin=None, save_path=None, dpi=300, **kwargs):
    """
    可视化指定区域下的 Metacell Hi-C 热图 (以网格形式展示)。
    
    参数:
    - hdata: 核心数据对象
    - chrom: 染色体名称
    - start: 起始位置
    - end: 终止位置
    - resolution: 分辨率
    - celltype: 目标细胞类型名称。如果指定，则只画该细胞类型的 Metacells；如果不指定，画所有 Metacells。
    - base_on: 'pair' (通过 mcool 读取) 或 'mat' (直接读取 views_mat 聚合结果)
    - balance: 是否使用 balance 后的矩阵
    - ncols: 网格每一行展示的图表数量
    - named_on: 如果指定，则使用 hdata.metacells 中的该列属性来命名子图标题
    - cell_type_col: hdata.metacells 中记录细胞类型的列名，默认为 'cell_type'
    """
    # 1. 获取要绘制的 Metacell IDs
    if celltype is not None:
        if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
            raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
            
        target_obs = hdata.metacells[hdata.metacells[cell_type_col] == celltype]
        m_ids = target_obs.index.tolist()
        if not m_ids:
            print(f"未找到细胞类型为 '{celltype}' 的 Metacell，请检查名称是否正确。")
            return
        title_prefix = f"Cell Type: {celltype}"
    else:
        # 画所有 metacells
        if hasattr(hdata, 'metacells'):
            m_ids = hdata.metacells.index.tolist()
        elif base_on == 'pair' and 'mcool' in hdata.metacell_data:
            m_ids = list(hdata.metacell_data['mcool'].keys())
        elif base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
            str_res = str(resolution)
            if base_on in hdata.metacell_data and str_res in hdata.metacell_data[base_on]:
                m_ids = list(hdata.metacell_data[base_on][str_res].keys())
            else:
                m_ids = []
        else:
            m_ids = []
            
        if not m_ids:
            print("未找到任何 Metacell 数据。")
            return
        title_prefix = "All Metacells"
        
    print(f"共找到 {len(m_ids)} 个 Metacells, 准备渲染...")
    
    if len(m_ids) > 100:
        print(f"警告：Metacell 数量 ({len(m_ids)}) 超过 100，绘制可能较慢。")
        
    # 2. 初始化网格画布
    nrows = math.ceil(len(m_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    
    if isinstance(axes, plt.Axes):
        axes = [axes] # 单个子图情况
    else:
        axes = axes.flatten() # 展平方便遍历
        
    # 3. 遍历渲染子图
    for i, m_id in enumerate(m_ids):
        ax = axes[i]
        ax.set_aspect('equal') # 强制子图内部坐标系严格正方形
        
        try:
            if base_on == 'pair':
                if m_id not in hdata.metacell_data.get('mcool', {}):
                    ax.set_title(f"{m_id}\n(No mcool data)")
                    ax.axis('off')
                    continue
                mcool_path = hdata.metacell_data['mcool'][m_id]
                uri = f"{mcool_path}::/resolutions/{resolution}"
                clr = cooler.Cooler(uri)
                mat = clr.matrix(balance=balance).fetch((chrom, start, end))
                
            elif base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
                str_res = str(resolution)
                dict_key = base_on
                if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
                    raise ValueError(f"未找到基于 {base_on} 聚合的分辨率数据。")
                mcool_dict = hdata.metacell_data[dict_key][str_res]
                if m_id not in mcool_dict or chrom not in mcool_dict[m_id]:
                    ax.set_title(f"{m_id}\n(No {base_on} data)")
                    ax.axis('off')
                    continue
                
                whole_chrom_mat = mcool_dict[m_id][chrom]
                start_bin = int(start // resolution)
                end_bin = int(np.ceil(end / resolution))
                max_bins = whole_chrom_mat.shape[0]
                start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)
                
                import scipy.sparse as sp
                if sp.issparse(whole_chrom_mat):
                    mat = whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
                else:
                    mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]
            else:
                raise ValueError("base_on 必须是 'pair' 或 'mat'")

            mat = np.nan_to_num(mat)
            if log1p:
                mat = np.log1p(mat)
            if fill_diagonal_zero:
                np.fill_diagonal(mat, 0)

            if vmax is None:
                if vmin is None:
                    sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax)
                else:
                    sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin)
            else:
                if vmin is None:
                    sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmax=vmax)
                else:
                    sns.heatmap(mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax)
            if named_on is not None and hasattr(hdata, 'metacells') and named_on in hdata.metacells.columns:
                m_name = hdata.metacells.loc[m_id, named_on]
                ax.set_title(f"{m_id} ({m_name})")
            else:
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
    plt.suptitle(f"{title_prefix} | Region: {chrom}:{start}-{end} @ {resolution//1000}kb", 
                 y=1.02, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# ==========================================================
# 3. 结构增强版可视化 (First-Principle O/E 校正)
# ==========================================================

from scipy.ndimage import gaussian_filter1d

def _calculate_oe(mat, log2_transform=True, pseudocount=1.0, mask_threshold=0.05):
    """
    计算 Observed/Expected (O/E) 矩阵核心算法。
    已增设[高斯平滑距离衰减曲线]与[低覆盖坏点遮蔽]功能，彻底消除由于稀疏数据带来的平行对角线锯齿条纹与贯穿式十字蓝带。
    """
    n = mat.shape[0]
    
    # 1. 坏点检测 (Coverage极低的测序死角 unmappable bins，必须拉黑，否则算 O/E 会被放大为贯穿的蓝色十字条纹)
    cov = np.sum(mat, axis=1) + np.sum(mat, axis=0)
    bad_bins = cov < (np.mean(cov) * mask_threshold)
    
    # 2. 计算 1D 物理期望衰减曲线 (因为单细胞稀疏性，如果不加修饰直接用，远距离全是在剧烈抖动的锯齿！)
    raw_expected_curve = np.zeros(n)
    for d in range(n):
        diag_val = np.diag(mat, d)
        valid_vals = diag_val[diag_val > 0]
        if len(valid_vals) > 0:
            raw_expected_curve[d] = np.mean(valid_vals)
        else:
            raw_expected_curve[d] = 0.0
            
    # --- 核心破局修复点 ---
    # 单细胞级别的稀疏性，让 raw_expected_curve 在大基因组距离时数值忽上忽下（比如这一条对角线平均有2个点交互，下一条突然变成0，再下一条又变成3）。
    # 如果不强行把这条一维曲线抹平，这种抖动投射到二维就会变成【大量平行于主对角线的细条纹】！！！
    smoothed_curve = gaussian_filter1d(raw_expected_curve, sigma=3.0)
    smoothed_curve[smoothed_curve < 0] = 0 # 防止平滑出负数
            
    expected = np.zeros_like(mat, dtype=float)
    for d in range(n):
        val = smoothed_curve[d]
        np.fill_diagonal(expected[d:], val)
        if d != 0:
            np.fill_diagonal(expected[:, d:], val)
            
    # 计算带有假计数平滑的 (O + p) / (E + p)
    oe_mat = (mat + pseudocount) / (expected + pseudocount)
    
    # 3. 将前面检测到的基因组死角强行涂白（值设为 1.0，因为等会取了 log2 就会变成 0，在 RdBu 中就是纯白色的缝隙，而不碍眼）
    oe_mat[bad_bins, :] = 1.0
    oe_mat[:, bad_bins] = 1.0
    
    if log2_transform:
        oe_mat = np.log2(oe_mat)
        
    return oe_mat

def plot_metacell_heatmap_enhanced(hdata, metacell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='RdBu_r', vmin=-2, vmax=2, save_path=None, dpi=300, **kwargs):
    """
    【升级增强版】带 O/E 物理背景校正的单细胞/Metacell高分辨率 Hi-C 热图可视化。
    (保留原有数据读取功能，新增 O/E 过滤消除距离衰减红晕，让TAD和核心结构更清晰)
    """
    if base_on == 'pair':
        if 'mcool' not in hdata.metacell_data or metacell_id not in hdata.metacell_data['mcool']:
            raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mcool 记录。请确保已经运行过 aggregate_metacell_pairs。")
            
        mcool_path = hdata.metacell_data['mcool'][metacell_id]
        try:
            uri = f"{mcool_path}::/resolutions/{resolution}"
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在 mcool 文件中读取分辨率 {resolution}。确保 pair 流程初始化了该分辨率。详情: {e}")
            
    elif base_on in ['mat', 'mat_redist', 'mat_consensus', 'mat_EM']:
        str_res = str(resolution)
        dict_key = base_on
        if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
            raise ValueError(f"未找到基于 {base_on} 聚合的分辨率 {resolution} 数据。请先运行对应的聚合函数。")
            
        mcool_dict = hdata.metacell_data[dict_key][str_res]
        
        # 数据类型的强健性处理：容忍用户传入 int 但字典里是 str，或反之
        if metacell_id not in mcool_dict and str(metacell_id) in mcool_dict:
            metacell_id = str(metacell_id)
        elif type(metacell_id) is str and metacell_id.isdigit() and int(metacell_id) in mcool_dict:
            metacell_id = int(metacell_id)
            
        if metacell_id not in mcool_dict:
             raise ValueError(f"未找到 Metacell '{metacell_id}' 的 mat 聚合记录。当前内存中已有的 ID 包括: {list(mcool_dict.keys())}")
             
        if chrom not in mcool_dict[metacell_id]:
             raise ValueError(f"未找到染色体 {chrom} 的聚合矩阵。")
             
        whole_chrom_mat = mcool_dict[metacell_id][chrom]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) 
    
    # ======== 核心结构升级步骤 ========
    # 使用 O/E 矩阵校正算法替代原本的 log1p，彻底消除背景压制
    oe_mat = _calculate_oe(mat, log2_transform=True)
    oe_mat = np.nan_to_num(oe_mat)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # O/E 矩阵使用发散渐变色以 0 为中点 (红=高于预期，蓝=低于预期)
    sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
    
    ax.set_title(f"Metacell (O/E Enhanced): {metacell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return oe_mat

def plot_cell_of_metacell_heatmap_enhanced(hdata, metacell_id, cell_id, chrom, start, end, resolution, balance=True, base_on='pair', cmap='RdBu_r', vmin=-2, vmax=2, save_path=None, dpi=300, **kwargs):
    """
    【升级增强版】带 O/E 物理背景校正的指定 Metacell 内单个原始细胞的高分辨率 Hi-C 热图可视化。
    """
    mat = None
    if base_on == 'pair':
        cell_name = str(cell_id)
        if isinstance(cell_id, int) or str(cell_id).isdigit():
            idx = int(cell_id)
            all_files = []
            import os
            for val in sorted(os.listdir(hdata.data_dir)):
                if val.endswith('.pairs') or val.endswith('.pairs.gz'):
                    all_files.append(val)
            if idx < len(all_files):
                cell_name = all_files[idx].split('.pairs')[0]
        
        mcool_path = None
        import os
        search_dirs = [hdata.data_dir, hdata.output_dir]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    if f.endswith('.mcool') or f.endswith('.cool'):
                        if cell_name in f or str(cell_id) in f:
                            mcool_path = os.path.join(root, f)
                            break
                if mcool_path: break
            if mcool_path: break
            
        if not mcool_path:
            raise ValueError(f"未找到单细胞 '{cell_id}' 对应的 .mcool 或 .cool 文件，请确认已生成该文件或使用 base_on='mat'。")
            
        try:
            import cooler
            if mcool_path.endswith('.mcool'):
                uri = f"{mcool_path}::/resolutions/{resolution}"
            else:
                uri = f"{mcool_path}::/resolutions/{resolution}"
                if not cooler.fileops.is_multires_file(mcool_path):
                    uri = mcool_path
            clr = cooler.Cooler(uri)
            mat = clr.matrix(balance=balance).fetch((chrom, start, end))
        except Exception as e:
            raise ValueError(f"无法在文件 {mcool_path} 中读取数据。详情: {e}")
            
    elif base_on == 'mat' or base_on == 'mat_redist':
        if not hdata.views_mat:
            raise ValueError("hdata.views_mat 为空，请确认已运行预处理。")
        _vk = _resolve_view_key(hdata, resolution, 'views_mat')
        if chrom not in hdata.views_mat[_vk]:
            raise ValueError(f"未找到染色体 {chrom} 的 views_mat 数据。")
            
        try:
            cell_idx = hdata.obs.index.get_loc(cell_id)
        except KeyError:
            if isinstance(cell_id, int) and cell_id < len(hdata.obs):
                cell_idx = cell_id
            else:
                raise ValueError(f"Cell ID '{cell_id}' 不在 hdata.obs.index 中。")
                
        whole_chrom_mat = hdata.views_mat[_vk][chrom][cell_idx]
        start_bin = int(start // resolution)
        end_bin = int(np.ceil(end / resolution))
        max_bins = whole_chrom_mat.shape[0]
        start_bin = max(0, start_bin)
        end_bin = min(max_bins, end_bin)
        
        import scipy.sparse as sp
        if sp.issparse(whole_chrom_mat):
            mat = whole_chrom_mat.tocsr()[start_bin:end_bin, start_bin:end_bin].toarray()
        else:
            mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin]
    else:
        raise ValueError("base_on 参数必须是 'pair' 或 'mat'")

    mat = np.nan_to_num(mat) 
    oe_mat = _calculate_oe(mat, log2_transform=True)
    oe_mat = np.nan_to_num(oe_mat)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
    
    ax.set_title(f"Metacell (O/E Enhanced): {metacell_id} | Cell: {cell_id}\n{chrom}:{start}-{end} @ {resolution//1000}kb", pad=15)
    ax.set_xlabel("Genomic Bins")
    ax.set_ylabel("Genomic Bins")
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return oe_mat

def plot_celltype_heatmaps_enhanced(hdata, cell_type, chrom, start, end, resolution, 
                             balance=True, base_on='pair', ncols=4, 
                             cell_type_col='cell_type', cmap='RdBu_r',
                              vmin=-2, vmax=2, save_path=None, dpi=300):
    """
    【升级增强版】可视化指定细胞类型下所有 Metacell 的 O/E 校正 Hi-C 热图 (网格展板)。
    """
    if not hasattr(hdata, 'metacells') or cell_type_col not in hdata.metacells.columns:
        raise ValueError(f"hdata.metacells 中未找到列 '{cell_type_col}'，请确认存放细胞类型的列名。")
        
    target_obs = hdata.metacells[hdata.metacells[cell_type_col] == cell_type]
    m_ids = target_obs.index.tolist()
    
    if not m_ids:
        print(f"未找到细胞类型为 '{cell_type}' 的 Metacell，请检查名称是否正确。")
        return
        
    print(f"共找到 {len(m_ids)} 个属于 '{cell_type}' 的 Metacells, 准备渲染带有物理衰减校正增强的热图...")
    
    nrows = math.ceil(len(m_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, m_id in enumerate(m_ids):
        ax = axes[i]
        ax.set_aspect('equal')
        
        try:
            if base_on == 'pair':
                if m_id not in hdata.metacell_data.get('mcool', {}):
                    ax.set_title(f"{m_id}\n(No mcool data)")
                    ax.axis('off')
                    continue
                mcool_path = hdata.metacell_data['mcool'][m_id]
                uri = f"{mcool_path}::/resolutions/{resolution}"
                clr = cooler.Cooler(uri)
                mat = clr.matrix(balance=balance).fetch((chrom, start, end))
                
            elif base_on in ['mat', 'mat_redist']:
                str_res = str(resolution)
                dict_key = 'mat' if base_on == 'mat' else 'mat_redist'
                if dict_key not in hdata.metacell_data or str_res not in hdata.metacell_data[dict_key]:
                    raise ValueError(f"未找到基于 {base_on} 聚合的分辨率数据。")
                mcool_dict = hdata.metacell_data[dict_key][str_res]
                if m_id not in mcool_dict or chrom not in mcool_dict[m_id]:
                    ax.set_title(f"{m_id}\n(No mat data)")
                    ax.axis('off')
                    continue
                
                whole_chrom_mat = mcool_dict[m_id][chrom]
                start_bin = int(start // resolution)
                end_bin = int(np.ceil(end / resolution))
                max_bins = whole_chrom_mat.shape[0]
                start_bin, end_bin = max(0, start_bin), min(max_bins, end_bin)
                mat = whole_chrom_mat[start_bin:end_bin, start_bin:end_bin].toarray()
            else:
                raise ValueError("base_on 必须是 'pair' 或 'mat'")

            mat = np.nan_to_num(mat)
            
            # ======== 核心升级步骤 ========
            oe_mat = _calculate_oe(mat, log2_transform=True)
            oe_mat = np.nan_to_num(oe_mat)

            sns.heatmap(oe_mat, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, center=0.0)
            
            ax.set_title(m_id)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.set_title(f"{m_id}\n(Error)")
            ax.axis('off')
            
    for j in range(len(m_ids), len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"[O/E Enhanced] Cell Type: {cell_type} | Region: {chrom}:{start}-{end} @ {resolution//1000}kb", 
                 y=1.02, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if 'save_path' in locals() and save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()