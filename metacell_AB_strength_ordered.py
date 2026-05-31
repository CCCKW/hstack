"""
metacell_AB_strength_ordered.py
================================
按 sorted_metacells 的 pseudotime 排序（来自 metacell_cycle_heatmap_order.csv），
将每个 metacell 的 compartment strength 画成散点图 + 平滑折线。

与 metacell_AB_strength.py 的区别
-----------------------------------
- 旧脚本：按 cell cycle 阶段分组取均值，x 轴是 phase 标签
- 本脚本：x 轴是 pseudotime rank（每个 metacell 一个点），颜色区分 phase

用法
----
# 直接使用已有的两个 CSV（最快）：
python metacell_AB_strength_ordered.py \
    --order_csv metacell_cycle_heatmap_order.csv \
    --strength_csv compartment_strength.csv \
    --output compartment_strength_ordered.pdf

# 或者重新从 h5ad 计算 strength：
python metacell_AB_strength_ordered.py \
    --order_csv metacell_cycle_heatmap_order.csv \
    --hdata cycle_h5ad/cycle.h5ad \
    --chroms chr11 --resolution hic_1000000 \
    --output compartment_strength_ordered.pdf
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- 颜色配置（与 notebook 一致） ----------
PHASE_ORDER  = ["post-M", "G1", "early-S", "late-S-G2", "pre-M"]
PHASE_COLORS = {
    "post-M":    "red",
    "G1":        "blue",
    "early-S":   "green",
    "late-S-G2": "purple",
    "pre-M":     "orange",
}

# ==================== 计算 strength 的辅助函数 ====================

def _to_csr(mat):
    if isinstance(mat, csr_matrix):
        return mat
    return csr_matrix(mat)


def _observed_over_expected(dense: np.ndarray) -> np.ndarray:
    n = dense.shape[0]
    oe = np.zeros_like(dense, dtype=np.float64)
    for d in range(n):
        idx = np.arange(n - d)
        diag = dense[idx, idx + d]
        m = diag[diag > 0].mean() if np.any(diag > 0) else 0.0
        vals = diag / m if m > 0 else diag
        oe[idx, idx + d] = vals
        oe[idx + d, idx] = vals
    return oe


def _compartment_eigvec(oe: np.ndarray, coverage: np.ndarray):
    with np.errstate(divide="ignore"):
        logoe = np.log2(oe)
    logoe[~np.isfinite(logoe)] = 0.0
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
    pc1 = eigvecs[:, -1]
    full = np.full(coverage.shape[0], np.nan, dtype=np.float64)
    full[valid] = pc1
    return full, valid


def calc_compartment_strength(mat, n_bins: int = 5) -> float:
    dense = _to_csr(mat).toarray().astype(np.float64)
    coverage = dense.sum(axis=1)
    oe = _observed_over_expected(dense)
    pc1, valid = _compartment_eigvec(oe, coverage)
    if pc1 is None:
        return np.nan
    v = pc1[valid]
    sub_oe = oe[np.ix_(valid, valid)]
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


def compute_strength_from_hdata(h, order_df: pd.DataFrame,
                                 chroms: list, resolution: str) -> pd.DataFrame:
    records = []
    for _, row in order_df.iterrows():
        mc_id = int(row["metacell"])
        for chrom in chroms:
            try:
                mat = h.metacell_data["mat"][resolution][mc_id][chrom]
            except KeyError:
                continue
            s = calc_compartment_strength(mat)
            records.append({"metacell": mc_id, "cell_cycle": row["cell_cycle"],
                             "chrom": chrom, "strength": s})
    return pd.DataFrame(records)


# ==================== 绘图 ====================

def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """简单移动平均平滑，忽略 NaN。"""
    out = np.full_like(y, np.nan, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return out
    tmp = y.copy().astype(float)
    tmp[~valid] = np.nanmean(y)      # 用均值填 NaN 再平滑，避免边缘畸变
    smoothed = uniform_filter1d(tmp, size=window, mode="nearest")
    out[valid] = smoothed[valid]
    return out


def plot_ordered(order_df: pd.DataFrame, strength_df: pd.DataFrame,
                 smooth_window: int, output: str, dpi: int):
    """
    order_df  columns: metacell, cell_cycle, order
    strength_df columns: metacell, cell_cycle, chrom, strength
              OR         metacell, cell_cycle, strength  (已聚合)
    """
    # --- 聚合多条染色体到每个 metacell 的均值 ---
    if "chrom" in strength_df.columns:
        mc_strength = (
            strength_df.groupby("metacell")["strength"]
            .mean()
            .reset_index()
            .rename(columns={"strength": "mean_strength"})
        )
    else:
        mc_strength = strength_df.rename(columns={"strength": "mean_strength"})[
            ["metacell", "mean_strength"]
        ]

    # --- 合并排序信息 ---
    merged = order_df.merge(mc_strength, on="metacell", how="left")
    merged = merged.sort_values("order").reset_index(drop=True)

    x      = merged["order"].to_numpy()
    y      = merged["mean_strength"].to_numpy(dtype=float)
    phases = merged["cell_cycle"].tolist()
    colors = [PHASE_COLORS.get(p, "#888888") for p in phases]

    # --- 平滑曲线 ---
    y_smooth = _smooth(y, window=smooth_window)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)

    # 背景色块：标记各 phase 区域
    phase_segs = []
    cur_phase, seg_start = phases[0], x[0]
    for i in range(1, len(phases)):
        if phases[i] != cur_phase:
            phase_segs.append((cur_phase, seg_start, x[i - 1]))
            cur_phase, seg_start = phases[i], x[i]
    phase_segs.append((cur_phase, seg_start, x[-1]))

    ymin, ymax = np.nanmin(y) - 0.05, np.nanmax(y) + 0.05
    for ph, xs, xe in phase_segs:
        c = PHASE_COLORS.get(ph, "#888888")
        ax.axvspan(xs - 0.5, xe + 0.5, color=c, alpha=0.07, lw=0)

    # 散点
    ax.scatter(x, y, c=colors, s=30, zorder=3, alpha=0.85, edgecolors="none")

    # 平滑均值线
    valid_mask = np.isfinite(y_smooth)
    if valid_mask.sum() > 1:
        ax.plot(x[valid_mask], y_smooth[valid_mask],
                color="black", lw=2, zorder=4, label=f"smooth (w={smooth_window})")

    # --- 相位分隔线 ---
    for _, xs, xe in phase_segs[:-1]:
        ax.axvline(xe + 0.5, color="gray", lw=0.8, ls="--", alpha=0.6)

    # --- 图例 ---
    present_phases = [p for p in PHASE_ORDER if p in set(phases)]
    # 若 PHASE_ORDER 中没有匹配项（如 "unknown"），回退到实际出现的 phase
    if not present_phases:
        present_phases = sorted(set(phases))
    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS.get(p, "#888888"), label=p)
        for p in present_phases
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, frameon=False,
                  loc="upper right", ncol=max(1, len(legend_handles)))

    ax.set_xlabel("Pseudotime rank (sorted_metacells order)", fontsize=10)
    ax.set_ylabel("Compartment strength\nlog2((AA+BB)/(AB+BA))", fontsize=10)
    ax.set_title("Compartment strength by pseudotime order", fontsize=11)
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {output}")


# ==================== main ====================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot compartment strength ordered by pseudotime (sorted_metacells)."
    )
    p.add_argument("--order_csv", required=True,
                   help="metacell_cycle_heatmap_order.csv（含 metacell, cell_cycle, order 列）")
    # 强度来源：二选一
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--strength_csv",
                   help="已有的 compartment_strength.csv（含 metacell, strength[, chrom] 列）")
    g.add_argument("--hdata",
                   help="从 h5ad 重新计算 strength")
    # 仅 --hdata 时需要
    p.add_argument("--chroms", nargs="+", default=["chr11"])
    p.add_argument("--resolution", default="hic_1000000")
    p.add_argument("--output_csv", default=None,
                   help="计算结果另存 CSV（仅 --hdata 时有效）")
    # 通用
    p.add_argument("--smooth_window", type=int, default=3,
                   help="平滑窗口大小（奇数，1=不平滑）")
    p.add_argument("--output", default="compartment_strength_ordered.pdf")
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    order_df = pd.read_csv(args.order_csv)
    # 兼容只有 metacell_id 一列的简单格式（行顺序即排序）
    if "metacell_id" in order_df.columns and "metacell" not in order_df.columns:
        order_df = order_df.rename(columns={"metacell_id": "metacell"})
    if "order" not in order_df.columns:
        order_df["order"] = range(len(order_df))
    if "cell_cycle" not in order_df.columns:
        order_df["cell_cycle"] = "unknown"
    for col in ("metacell", "cell_cycle", "order"):
        if col not in order_df.columns:
            raise ValueError(f"order_csv 缺少列: {col}")
    order_df["metacell"] = order_df["metacell"].astype(int)

    if args.strength_csv:
        strength_df = pd.read_csv(args.strength_csv)
        if "metacell" not in strength_df.columns or "strength" not in strength_df.columns:
            raise ValueError("strength_csv 需含 'metacell' 和 'strength' 列")
        strength_df["metacell"] = strength_df["metacell"].astype(int)
    else:
        import stark as sk
        print(f"读取 h5ad: {args.hdata}")
        h = sk.HData.read_h5ad(args.hdata)
        sk.tl.aggregate_metacell_mat(h, force_aggregate=False)
        strength_df = compute_strength_from_hdata(h, order_df, args.chroms, args.resolution)
        if args.output_csv:
            strength_df.to_csv(args.output_csv, index=False)
            print(f"strength CSV 已保存: {args.output_csv}")

    plot_ordered(order_df, strength_df,
                 smooth_window=args.smooth_window,
                 output=args.output,
                 dpi=args.dpi)
