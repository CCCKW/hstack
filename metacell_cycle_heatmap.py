"""
metacell_cycle_heatmap.py
=========================
按细胞周期顺序排列 metacell，计算 metacell 两两 SCC（分层相关），并绘制热图。

依赖数据结构（与 compare_scc.py 一致）
------------------------------------
* h5ad 中需要有 metacells 表，并包含周期列（默认 CellType）
* 可由 stark.tl.aggregate_metacell_mat 生成 metacell_data['mat']

示例
----
python metacell_cycle_heatmap.py \
  --hdata cycle_10k.h5ad \
    --chroms chr11 chr12 chr13 \
  --resolution 1000000 \
  --cell_cycles G1 early-S late-S-G2 \
  --output heatmap_cycle_scc.pdf
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _to_csr(mat):
    if isinstance(mat, csr_matrix):
        return mat
    return csr_matrix(mat)


def calc_stratum_cor(
    mat1,
    mat2,
    max_dist: int = 50,
    depth_normalize: bool = True,
    use_log1p: bool = True,
) -> float:
    """Compute SCC-like stratum correlation used in compare_scc.py."""
    mat1, mat2 = _to_csr(mat1), _to_csr(mat2)

    if depth_normalize:
        sum1, sum2 = float(mat1.sum()), float(mat2.sum())
        if sum1 <= 0 or sum2 <= 0:
            return 0.0
    else:
        sum1 = sum2 = 1.0

    cor_strata, weight_strata = [], []
    for d in range(1, max_dist + 1):
        v1 = mat1.diagonal(d).astype(np.float64, copy=False)
        v2 = mat2.diagonal(d).astype(np.float64, copy=False)
        if len(v1) <= 1:
            continue

        if depth_normalize:
            v1 = v1 / sum1 * 1e6
            v2 = v2 / sum2 * 1e6
        if use_log1p:
            v1 = np.log1p(v1)
            v2 = np.log1p(v2)

        var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)
        if var1 == 0 or var2 == 0:
            continue

        corr, _ = pearsonr(v1, v2)
        if np.isnan(corr):
            continue

        weight = np.sqrt(var1 * var2) * len(v1)
        cor_strata.append(corr)
        weight_strata.append(weight)

    if not weight_strata:
        return 0.0
    return float(np.average(cor_strata, weights=weight_strata))


def order_metacells_by_cycle(metacells: pd.DataFrame, cycle_order: list[str], celltype_col: str):
    if celltype_col not in metacells.columns:
        raise KeyError(f"metacells 缺少列: {celltype_col}")

    seen = set(cycle_order)
    present_cycles = metacells[celltype_col].astype(str).tolist()
    remain = [c for c in pd.unique(present_cycles).tolist() if c not in seen]
    final_cycles = cycle_order + remain

    ordered_ids = []
    ordered_cycle = []
    for cyc in final_cycles:
        sub = metacells[metacells[celltype_col].astype(str) == cyc]
        if len(sub) == 0:
            continue
        ids = sub.index.tolist()
        ordered_ids.extend(ids)
        ordered_cycle.extend([cyc] * len(ids))

    return ordered_ids, ordered_cycle


def build_scc_matrix(mc_mats: list, max_dist: int) -> np.ndarray:
    n = len(mc_mats)
    out = np.eye(n, dtype=np.float64)
    total = n * (n - 1) // 2
    pbar = tqdm(total=total, desc="Pairwise SCC", leave=False)
    for i in range(n):
        for j in range(i + 1, n):
            c = calc_stratum_cor(mc_mats[i], mc_mats[j], max_dist=max_dist)
            out[i, j] = c
            out[j, i] = c
            pbar.update(1)
    pbar.close()
    return out


def collect_mats_for_chrom(
    h,
    ordered_ids: list,
    ordered_cycles: list,
    resolution: int,
    chrom: str,
):
    mats = []
    valid_ids = []
    valid_cycles = []
    for mc_id, cyc in zip(ordered_ids, ordered_cycles):
        try:
            mat = h.metacell_data["mat"][resolution][mc_id][chrom]
        except KeyError:
            continue
        mats.append(mat)
        valid_ids.append(mc_id)
        valid_cycles.append(cyc)
    return valid_ids, valid_cycles, mats


def _fiedler_order_from_corr(corr: np.ndarray) -> np.ndarray:
    """Return index order based on Fiedler vector of Laplacian from correlation."""
    n = corr.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    # Keep only non-negative affinity to build a stable graph Laplacian.
    w = np.clip(corr, 0.0, None).copy()
    np.fill_diagonal(w, 0.0)
    deg = np.sum(w, axis=1)
    lap = np.diag(deg) - w

    try:
        eigvals, eigvecs = np.linalg.eigh(lap)
    except np.linalg.LinAlgError:
        return np.arange(n, dtype=int)

    if eigvecs.shape[1] < 2:
        return np.arange(n, dtype=int)

    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)


def reorder_within_cycle(ids: list, cycles: list, corr: np.ndarray):
    """Keep cycle block order, but optimize order inside each cycle block."""
    cycles = [str(x) for x in cycles]
    n = len(ids)
    if n <= 2:
        return ids, cycles, corr

    final_perm = []
    start = 0
    while start < n:
        cyc = cycles[start]
        end = start + 1
        while end < n and cycles[end] == cyc:
            end += 1

        block_idx = np.arange(start, end, dtype=int)
        if len(block_idx) > 2:
            sub = corr[np.ix_(block_idx, block_idx)]
            local_order = _fiedler_order_from_corr(sub)
            block_idx = block_idx[local_order]
        final_perm.extend(block_idx.tolist())
        start = end

    final_perm = np.asarray(final_perm, dtype=int)
    new_ids = [ids[i] for i in final_perm]
    new_cycles = [cycles[i] for i in final_perm]
    new_corr = corr[np.ix_(final_perm, final_perm)]
    return new_ids, new_cycles, new_corr


def _auto_figsize(n: int) -> tuple[float, float]:
    side = max(6.0, min(16.0, 4.0 + n * 0.12))
    return side, side


def plot_heatmap(
    corr_df: pd.DataFrame,
    cycle_labels: list[str],
    output: str,
    dpi: int,
    cmap: str,
    robust_scale: bool,
):
    n = corr_df.shape[0]
    fig_w, fig_h = _auto_figsize(n)

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    if robust_scale:
        vals = corr_df.to_numpy(dtype=float)
        tri = vals[np.triu_indices_from(vals, k=1)]
        tri = tri[np.isfinite(tri)]
        if len(tri) > 10:
            vmin = float(np.quantile(tri, 0.02))
            vmax = float(np.quantile(tri, 0.98))
            if vmin >= vmax:
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = -1.0, 1.0

    sns.heatmap(
        corr_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "SCC"},
        ax=ax,
    )

    cycle_labels = [str(x) for x in cycle_labels]
    run = []
    if cycle_labels:
        start = 0
        cur = cycle_labels[0]
        for i in range(1, len(cycle_labels) + 1):
            if i == len(cycle_labels) or cycle_labels[i] != cur:
                run.append((cur, start, i))
                if i < len(cycle_labels):
                    start = i
                    cur = cycle_labels[i]

    centers = []
    names = []
    for cyc, st, ed in run:
        centers.append((st + ed) / 2.0)
        names.append(cyc)
        if st > 0:
            ax.axhline(st, color="white", lw=1.2)
            ax.axvline(st, color="white", lw=1.2)

    if centers:
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)

    ax.set_title("Metacell Pairwise SCC Heatmap (Cell-Cycle Ordered)")
    ax.set_xlabel("Metacell (ordered by cell cycle)")
    ax.set_ylabel("Metacell (ordered by cell cycle)")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run(
    hdata_path: str,
    chroms: list[str],
    resolution: int,
    max_dist: int,
    cell_cycles: list[str],
    output: str,
    output_csv: str,
    output_order_csv: str,
    output_chrom_stats_csv: str,
    celltype_col: str,
    dpi: int,
    cmap: str,
    order_mode: str,
    robust_scale: bool,
):
    import stark as sk

    print(f"读取 h5ad: {hdata_path}")
    h = sk.HData.read_h5ad(hdata_path)
    sk.tl.aggregate_metacell_mat(h, force_aggregate=False)

    ordered_ids, ordered_cycles = order_metacells_by_cycle(
        h.metacells, cycle_order=cell_cycles, celltype_col=celltype_col
    )
    if len(ordered_ids) < 2:
        raise ValueError("可用 metacell 数量 < 2，无法计算两两相关性")

    if len(chroms) == 0:
        raise ValueError("至少需要一个染色体（--chrom 或 --chroms）")

    chrom_maps = {}
    for chrom in chroms:
        c_ids, c_cycles, c_mats = collect_mats_for_chrom(
            h=h,
            ordered_ids=ordered_ids,
            ordered_cycles=ordered_cycles,
            resolution=resolution,
            chrom=chrom,
        )
        if len(c_mats) < 2:
            print(f"[skip] {chrom}: 可用 metacell < 2")
            continue
        chrom_maps[chrom] = {
            "id_to_mat": {mc_id: mat for mc_id, mat in zip(c_ids, c_mats)},
            "id_to_cycle": {mc_id: cyc for mc_id, cyc in zip(c_ids, c_cycles)},
            "ids": c_ids,
        }

    if len(chrom_maps) == 0:
        raise ValueError("所有指定染色体都不可用，请检查 --resolution 与染色体名称")

    usable_chroms = list(chrom_maps.keys())
    id_sets = [set(chrom_maps[c]["ids"]) for c in usable_chroms]
    common_ids = [mc_id for mc_id in ordered_ids if all(mc_id in s for s in id_sets)]
    if len(common_ids) < 2:
        raise ValueError("多染色体共同可用的 metacell 数量 < 2，无法计算平均热图")

    valid_ids = common_ids
    valid_cycles = [str(chrom_maps[usable_chroms[0]]["id_to_cycle"][mc_id]) for mc_id in valid_ids]

    corr_list = []
    chrom_stats = []
    for chrom in usable_chroms:
        mats = [chrom_maps[chrom]["id_to_mat"][mc_id] for mc_id in valid_ids]
        print(f"计算 SCC 矩阵: {chrom} (n_metacell={len(mats)})")
        cmat = build_scc_matrix(mats, max_dist=max_dist)
        corr_list.append(cmat)

        tri = cmat[np.triu_indices_from(cmat, k=1)]
        tri = tri[np.isfinite(tri)]
        chrom_stats.append(
            {
                "chrom": chrom,
                "n_metacell": len(mats),
                "n_pairs": int(len(tri)),
                "mean_offdiag_scc": float(np.mean(tri)) if len(tri) else np.nan,
                "std_offdiag_scc": float(np.std(tri)) if len(tri) else np.nan,
            }
        )

    corr = np.mean(np.stack(corr_list, axis=0), axis=0)

    if order_mode == "within_cycle_fiedler":
        print("应用组内排序: within_cycle_fiedler")
        valid_ids, valid_cycles, corr = reorder_within_cycle(valid_ids, valid_cycles, corr)

    labels = [str(x) for x in valid_ids]
    corr_df = pd.DataFrame(corr, index=labels, columns=labels)
    corr_df.to_csv(output_csv)

    order_df = pd.DataFrame({
        "metacell": labels,
        "cell_cycle": valid_cycles,
        "order": np.arange(len(labels), dtype=int),
    })
    order_df.to_csv(output_order_csv, index=False)
    pd.DataFrame(chrom_stats).to_csv(output_chrom_stats_csv, index=False)

    plot_heatmap(
        corr_df=corr_df,
        cycle_labels=valid_cycles,
        output=output,
        dpi=dpi,
        cmap=cmap,
        robust_scale=robust_scale,
    )

    print(f"热图已保存: {output}")
    print(f"相关矩阵已保存: {output_csv}")
    print(f"排序信息已保存: {output_order_csv}")
    print(f"染色体统计已保存: {output_chrom_stats_csv}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute metacell pairwise SCC and draw cell-cycle ordered heatmap."
    )
    p.add_argument("--hdata", required=True, help="输入 h5ad 路径")
    p.add_argument("--chrom", default="chr11",
                   help="单染色体模式（兼容旧参数）；若同时给 --chroms，则优先使用 --chroms")
    p.add_argument("--chroms", nargs="+", default=None,
                   help="多染色体列表；会对每个染色体分别计算 SCC 矩阵并取平均")
    p.add_argument("--resolution", type=int, default=1_000_000)
    p.add_argument("--max_dist", type=int, default=50)
    p.add_argument("--cell_cycles", nargs="+", default=["G1", "early-S", "late-S-G2"])
    p.add_argument("--celltype_col", default="CellType")
    p.add_argument("--output", default="metacell_cycle_scc_heatmap.pdf")
    p.add_argument("--output_csv", default=None,
                   help="输出相关矩阵 CSV；默认由 --output 自动推导")
    p.add_argument("--output_order_csv", default=None,
                   help="输出 metacell 排序 CSV；默认由 --output 自动推导")
    p.add_argument("--output_chrom_stats_csv", default=None,
                   help="输出每个染色体的统计 CSV；默认由 --output 自动推导")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--cmap", default="vlag")
    p.add_argument("--order_mode", default="cycle_only", choices=["cycle_only", "within_cycle_fiedler"],
                   help="排序模式：cycle_only=仅按周期分块；within_cycle_fiedler=周期内再做一维排序")
    p.add_argument("--robust_scale", action="store_true",
                   help="使用分位数色阶（2%%-98%%）增强细微梯度显示")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out = Path(args.output)
    output_csv = args.output_csv or str(out.with_suffix(".csv"))
    output_order_csv = args.output_order_csv or str(out.with_name(out.stem + "_order.csv"))
    output_chrom_stats_csv = args.output_chrom_stats_csv or str(out.with_name(out.stem + "_chrom_stats.csv"))
    chroms = args.chroms if args.chroms else [args.chrom]

    run(
        hdata_path=args.hdata,
        chroms=chroms,
        resolution=args.resolution,
        max_dist=args.max_dist,
        cell_cycles=args.cell_cycles,
        output=str(out),
        output_csv=output_csv,
        output_order_csv=output_order_csv,
        output_chrom_stats_csv=output_chrom_stats_csv,
        celltype_col=args.celltype_col,
        dpi=args.dpi,
        cmap=args.cmap,
        order_mode=args.order_mode,
        robust_scale=args.robust_scale,
    )