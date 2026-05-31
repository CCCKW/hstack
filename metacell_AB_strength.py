"""
metacell_compartment_strength.py
================================
基于 metacell 的单染色体接触矩阵，计算 compartment strength（区室化强度），
并按细胞周期顺序（G1 -> early-S -> late-S-G2）绘制变化曲线。

方法
----
对每个矩阵：
  1. observed / expected 归一化（按 genomic distance 去趋势）
  2. 计算 Pearson correlation matrix
  3. 取 PC1 作为 A/B compartment eigenvector
  4. 按 PC1 值分箱做 saddle plot，strength = log2((AA+BB)/(AB+BA))

依赖数据结构（与 metacell_cycle_heatmap.py 一致）
------------------------------------------------
* h5ad 中需要有 metacells 表，并包含周期列（默认 dominant_label）
* 可由 stark.tl.aggregate_metacell_mat 生成 metacell_data['mat']

示例
----
python metacell_compartment_strength.py \
  --hdata cycle_10k.h5ad \
  --chroms chr11 chr12 chr13 \
  --resolution hic_1000000 \
  --cell_cycles G1 early-S late-S-G2 \
  --output compartment_strength.pdf
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _to_csr(mat):
    if isinstance(mat, csr_matrix):
        return mat
    return csr_matrix(mat)


def _observed_over_expected(dense: np.ndarray) -> np.ndarray:
    """按对角线（genomic distance）做 observed/expected 归一化。"""
    n = dense.shape[0]
    oe = np.zeros_like(dense, dtype=np.float64)
    for d in range(n):
        idx = np.arange(n - d)
        diag = dense[idx, idx + d]
        m = diag[diag > 0].mean() if np.any(diag > 0) else 0.0
        if m > 0:
            vals = diag / m
        else:
            vals = diag
        oe[idx, idx + d] = vals
        oe[idx + d, idx] = vals
    return oe


def _compartment_eigvec(oe: np.ndarray, coverage: np.ndarray):
    """从 O/E 矩阵得到 PC1（A/B eigenvector）。不做方向标定。"""
    # 用 log 让 plaid 模式更清晰；O/E 已 >0
    with np.errstate(divide="ignore"):
        logoe = np.log2(oe)
    logoe[~np.isfinite(logoe)] = 0.0

    # 只在有效 bin（coverage>0）上做相关
    valid = coverage > 0
    if valid.sum() < 4:
        return None, valid

    sub = logoe[np.ix_(valid, valid)]
    corr = np.corrcoef(sub)
    corr[~np.isfinite(corr)] = 0.0

    try:
        eigvals, eigvecs = np.linalg.eigh(corr)
    except np.linalg.LinAlgError:
        return None, valid
    pc1 = eigvecs[:, -1]  # 最大特征值对应向量；符号任意，strength 不依赖方向

    full = np.full(coverage.shape[0], np.nan, dtype=np.float64)
    full[valid] = pc1
    return full, valid


def calc_compartment_strength(mat, n_bins: int = 5) -> float:
    """单矩阵 compartment strength = log2((AA+BB)/(AB+BA))。

    A/B 按 PC1 符号划分（PC1>0 vs PC1<0），与方向标定无关：
    符号翻转时 A/B 互换、AA<->BB、AB<->BA，比值不变。
    n_bins 仅用于 saddle 分箱的可视化分辨率，不影响 strength 计算。
    """
    dense = _to_csr(mat).toarray().astype(np.float64)
    coverage = dense.sum(axis=1)

    oe = _observed_over_expected(dense)
    pc1, valid = _compartment_eigvec(oe, coverage)
    if pc1 is None:
        return np.nan

    v = pc1[valid]
    sub_oe = oe[np.ix_(valid, valid)]

    # 按 PC1 符号分成两组（不依赖方向）
    grp_a = v > 0
    grp_b = v < 0
    if grp_a.sum() < 1 or grp_b.sum() < 1:
        return np.nan

    def _block_mean(rows, cols):
        block = sub_oe[np.ix_(rows, cols)]
        block = block[block > 0]
        return block.mean() if len(block) else np.nan

    aa = _block_mean(grp_a, grp_a)
    bb = _block_mean(grp_b, grp_b)
    ab = _block_mean(grp_a, grp_b)
    ba = _block_mean(grp_b, grp_a)

    denom = ab + ba
    if not np.isfinite(denom) or denom <= 0 or not np.isfinite(aa + bb):
        return np.nan
    return float(np.log2((aa + bb) / denom))


def order_metacells_by_cycle(metacells: pd.DataFrame, cycle_order: list, celltype_col: str):
    if celltype_col not in metacells.columns:
        raise KeyError(f"metacells 缺少列: {celltype_col}")

    seen = set(cycle_order)
    present_cycles = metacells[celltype_col].astype(str).tolist()
    remain = [c for c in pd.unique(present_cycles).tolist() if c not in seen]
    final_cycles = cycle_order + remain

    ordered_ids, ordered_cycle = [], []
    for cyc in final_cycles:
        sub = metacells[metacells[celltype_col].astype(str) == cyc]
        if len(sub) == 0:
            continue
        ids = sub.index.tolist()
        ordered_ids.extend(ids)
        ordered_cycle.extend([cyc] * len(ids))
    return ordered_ids, ordered_cycle


def _sum_mats(mats: list):
    """把同一周期的多个稀疏矩阵相加得到聚合矩阵。"""
    acc = _to_csr(mats[0]).astype(np.float64).copy()
    for m in mats[1:]:
        acc = acc + _to_csr(m).astype(np.float64)
    return acc


def collect_mats_for_chrom(h, ordered_ids, ordered_cycles, resolution, chrom):
    mats, valid_ids, valid_cycles = [], [], []
    for mc_id, cyc in zip(ordered_ids, ordered_cycles):
        try:
            mat = h.metacell_data["mat"][resolution][int(mc_id)][chrom]
        except KeyError:
            continue
        mats.append(mat)
        valid_ids.append(mc_id)
        valid_cycles.append(cyc)
    return valid_ids, valid_cycles, mats


def run(
    hdata_path: str,
    chroms: list,
    resolution: str,
    cell_cycles: list,
    per: str,
    n_bins: int,
    output: str,
    output_csv: str,
    celltype_col: str,
    dpi: int,
):
    import stark as sk

    print(f"读取 h5ad: {hdata_path}")
    h = sk.HData.read_h5ad(hdata_path)
    sk.tl.aggregate_metacell_mat(h, force_aggregate=False)

    ordered_ids, ordered_cycles = order_metacells_by_cycle(
        h.metacells, cycle_order=cell_cycles, celltype_col=celltype_col
    )
    if len(chroms) == 0:
        raise ValueError("至少需要一个染色体（--chrom 或 --chroms）")

    records = []
    for chrom in chroms:
        c_ids, c_cycles, c_mats = collect_mats_for_chrom(
            h, ordered_ids, ordered_cycles, resolution, chrom
        )
        if len(c_mats) < 1:
            print(f"[skip] {chrom}: 无可用 metacell")
            continue

        if per == "cycle":
            # 按周期聚合后算一个 strength
            cyc_to_mats = {}
            for cyc, mat in zip(c_cycles, c_mats):
                cyc_to_mats.setdefault(cyc, []).append(mat)
            for cyc, mats in cyc_to_mats.items():
                agg = _sum_mats(mats)
                s = calc_compartment_strength(agg, n_bins=n_bins)
                records.append({"chrom": chrom, "cell_cycle": cyc,
                                "n_metacell": len(mats), "strength": s})
                print(f"  {chrom} | {cyc:12s} (n={len(mats):3d}) strength={s:.4f}")
        else:
            # 每个 metacell 单独算
            for mc_id, cyc, mat in zip(c_ids, c_cycles, c_mats):
                s = calc_compartment_strength(mat, n_bins=n_bins)
                records.append({"chrom": chrom, "metacell": mc_id,
                                "cell_cycle": cyc, "strength": s})

    if not records:
        raise ValueError("没有可用结果，请检查 --resolution 与染色体名称")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存: {output_csv}")

    _plot(df, cell_cycles, per, output, dpi)
    print(f"图已保存: {output}")


def _plot(df: pd.DataFrame, cell_cycles: list, per: str, output: str, dpi: int):
    # 确定周期 x 轴顺序
    present = [c for c in cell_cycles if c in set(df["cell_cycle"])]
    present += [c for c in pd.unique(df["cell_cycle"]) if c not in present]
    cyc_to_x = {c: i for i, c in enumerate(present)}

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)

    if per == "cycle":
        # 每条染色体一条线 + 均值
        for chrom, sub in df.groupby("chrom"):
            sub = sub.dropna(subset=["strength"])
            xs = [cyc_to_x[c] for c in sub["cell_cycle"]]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], sub["strength"].to_numpy()[order],
                    marker="o", alpha=0.4, lw=1, label=chrom)
        mean_df = df.dropna(subset=["strength"]).groupby("cell_cycle")["strength"].mean()
        xs = [cyc_to_x[c] for c in mean_df.index]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], mean_df.to_numpy()[order],
                marker="s", color="black", lw=2.5, label="mean")
        if df["chrom"].nunique() <= 8:
            ax.legend(fontsize=7, frameon=False)
    else:
        # metacell 级别：散点 + 周期均值
        df2 = df.dropna(subset=["strength"]).copy()
        jitter = (np.random.RandomState(0).rand(len(df2)) - 0.5) * 0.25
        xs = np.array([cyc_to_x[c] for c in df2["cell_cycle"]]) + jitter
        ax.scatter(xs, df2["strength"], s=10, alpha=0.4, color="#4477aa")
        mean_df = df2.groupby("cell_cycle")["strength"].mean()
        mxs = [cyc_to_x[c] for c in mean_df.index]
        order = np.argsort(mxs)
        ax.plot(np.array(mxs)[order], mean_df.to_numpy()[order],
                marker="s", color="black", lw=2.5, label="mean")
        ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel("Compartment strength  log2((AA+BB)/(AB+BA))")
    ax.set_xlabel("Cell cycle")
    ax.set_title("Compartment strength across cell cycle")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute metacell compartment strength across cell cycle."
    )
    p.add_argument("--hdata", required=True, help="输入 h5ad 路径")
    p.add_argument("--chrom", default="chr11", help="单染色体模式（兼容旧参数）")
    p.add_argument("--chroms", nargs="+", default=None, help="多染色体列表")
    p.add_argument("--resolution", default="hic_1000000")
    p.add_argument("--cell_cycles", nargs="+", default=["G1", "early-S", "late-S-G2"])
    p.add_argument("--celltype_col", default="dominant_label")
    p.add_argument("--per", default="metacell", choices=["cycle", "metacell"],
                   help="metacell=每个 metacell 单独算；cycle=按周期聚合算一个值")
    p.add_argument("--n_bins", type=int, default=5, help="saddle 分箱数量（quintile）")
    p.add_argument("--output", default="compartment_strength.pdf")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = Path(args.output)
    output_csv = args.output_csv or str(out.with_suffix(".csv"))
    chroms = args.chroms if args.chroms else [args.chrom]

    run(
        hdata_path=args.hdata,
        chroms=chroms,
        resolution=args.resolution,
        cell_cycles=args.cell_cycles,
        per=args.per,
        n_bins=args.n_bins,
        output=str(out),
        output_csv=output_csv,
        celltype_col=args.celltype_col,
        dpi=args.dpi,
    )