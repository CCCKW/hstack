"""
compare_scc.py
==============
对多种 metacell 算法计算分层相关性（SCC），绘制 SCI 风格小提琴图，
并在组内执行假设检验（Mann-Whitney U，双侧）。

数据假设
--------
* 第一个 h5ad（你的算法）必须包含：
    - views_mat[resolution][chrom]      原始单细胞接触矩阵（所有细胞）
    - metacell_data['mat'][resolution][mc_id][chrom]   已聚合的 metacell 矩阵
    - obs 列: label（细胞周期相标签）, metacell（cell→metacell 映射）
    - metacells 列: CellType（metacell 所属周期）

* 其余 h5ad 只需包含：
    - obs 列: label, metacell（分配方案，cell index 必须与第一个 h5ad 完全一致）
    - metacells 列: CellType
  （无需 views_mat / metacell_data['mat']，矩阵始终从第一个 h5ad 按各算法分组聚合）

* Single Cell 和 Random 基线均来自第一个 h5ad 的 views_mat，保证比较口径一致。

用法示例
--------
python compare_scc.py --hdatas my.h5ad seacells.h5ad metaq.h5ad \\
                      --names "Stark" "SEACells" "MetaQ" \\
                      --output comparison.pdf
"""

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 核心 SCC 计算
# ---------------------------------------------------------------------------

def _to_csr(mat):
    if isinstance(mat, csr_matrix):
        return mat
    return csr_matrix(mat)


def calc_stratum_cor(mat1, mat2, max_dist: int = 50,
                     depth_normalize: bool = True,
                     use_log1p: bool = True) -> float:
    """计算两个 Hi-C 矩阵的分层相关性（简化 SCC，含 CPM + log1p 深度归一化）。"""
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


def calc_pairwise_cor(mat_list, max_dist: int = 50, desc: str = "") -> list:
    n = len(mat_list)
    total = n * (n - 1) // 2
    return [
        calc_stratum_cor(a, b, max_dist=max_dist)
        for a, b in tqdm(itertools.combinations(mat_list, 2),
                         total=total, desc=desc, leave=False)
    ]


# ---------------------------------------------------------------------------
# 按分配方案聚合 metacell 矩阵
# ---------------------------------------------------------------------------

def build_metacell_matrices_from_assignment(
        obs_df, mc_ids, matrix_accessor, primary_obs_index,
        cycle: str, label_col: str, metacell_col: str,
        chrom: str, desc_prefix: str = ""):
    """
    根据 obs_df 中的 metacell 列分配，从 matrix_accessor 里聚合矩阵。
    只考虑 label==cycle 的细胞。
    返回 list of aggregated sparse matrices，顺序与 mc_ids 一致。
    """
    cycle_obs = obs_df[obs_df[label_col] == cycle]
    mc_matrices = []
    for mc_id in tqdm(mc_ids, desc=f"{desc_prefix} build MC mats ({cycle})", leave=False):
        members = cycle_obs[cycle_obs[metacell_col] == mc_id].index.tolist()
        if len(members) == 0:
            continue
        pos = [primary_obs_index.get_loc(cell) for cell in members
               if cell in primary_obs_index]
        if len(pos) == 0:
            continue
        agg = matrix_accessor[pos[0]].copy()
        for p in pos[1:]:
            agg = agg + matrix_accessor[p]
        mc_matrices.append(agg)
    return mc_matrices


# ---------------------------------------------------------------------------
# Random 伪 Metacell（无重叠分组）
# ---------------------------------------------------------------------------

def _match_sizes_to_pool(sizes, pool_size: int):
    sizes = np.asarray(sizes, dtype=int)
    sizes[sizes < 1] = 1
    total = int(sizes.sum())
    if total <= pool_size:
        return sizes.tolist()
    scaled = np.floor(sizes / total * pool_size).astype(int)
    scaled[scaled < 1] = 1
    while scaled.sum() > pool_size:
        idx = np.argmax(scaled)
        if scaled[idx] > 1:
            scaled[idx] -= 1
        else:
            break
    while scaled.sum() < pool_size:
        scaled[np.argmin(scaled)] += 1
    return scaled.tolist()


def build_random_pseudometacells(sc_pool_pos, group_sizes, matrix_accessor, rng):
    if len(sc_pool_pos) == 0:
        return []
    group_sizes = _match_sizes_to_pool(group_sizes, len(sc_pool_pos))
    shuffled = rng.permutation(sc_pool_pos)
    mats, start = [], 0
    for gsz in group_sizes:
        grp = shuffled[start:start + gsz]
        if len(grp) == 0:
            continue
        agg = matrix_accessor[grp[0]].copy()
        for idx in grp[1:]:
            agg = agg + matrix_accessor[idx]
        mats.append(agg)
        start += gsz
        if start >= len(shuffled):
            break
    return mats


# ---------------------------------------------------------------------------
# 统计辅助
# ---------------------------------------------------------------------------

def p_to_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_sig_bar(ax, x1, x2, y, h, text, lw=1.2):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _auto_palette(type_order):
    base = ["#2C7FB8", "#D95F02", "#7570B3", "#1B9E77", "#E7298A", "#66A61E",
            "#A6761D", "#666666"]
    algo_colors = (base * ((len(type_order)) // len(base) + 1))
    palette = {}
    algo_i = 0
    for t in type_order:
        if t == "Random":
            palette[t] = "#7FCDBB"
        elif t == "Single Cell":
            palette[t] = "#F03B20"
        else:
            palette[t] = algo_colors[algo_i]
            algo_i += 1
    return palette


def run(hdata_paths: list, names: list, cell_cycles: list,
        chrom: str, resolution: int, max_dist: int,
        n_sc: int, n_random: int, rng_seed: int,
        output: str, dpi: int,
        label_col: str = "label",
        metacell_col: str = "metacell",
        celltype_col: str = "CellType",
        random_scope: str = "cycle",
        plot_style: str = "violin"):
    """
    random_scope : "cycle"  — Random 只从当前周期的细胞中随机分组（默认）
                  "all"    — Random 从所有周期的细胞中随机分组
    plot_style   : "violin" — 小提琴图（默认）
                  "box"    — 箱线图
    """

    import stark as sk

    rng = np.random.default_rng(rng_seed)

    if not names:
        names = [Path(p).stem for p in hdata_paths]
    assert len(names) == len(hdata_paths), \
        f"--names 长度 ({len(names)}) 与 --hdatas 长度 ({len(hdata_paths)}) 不一致"

    primary_name = names[0]
    RANDOM_LABEL = "Random"
    SC_LABEL = "Single Cell"

    # ── 读取 h5ad ──
    print("正在读取 h5ad 文件...")
    hdatas = []
    for i, path in enumerate(hdata_paths):
        print(f"  [{names[i]}] {path}")
        h = sk.HData.read_h5ad(path)
        if i == 0:
            # 主算法：需要 aggregate 才能填充 metacell_data['mat']
            sk.tl.aggregate_metacell_mat(h, force_aggregate=False)
        hdatas.append(h)

    primary_hdata = hdatas[0]
    primary_obs_index = primary_hdata.obs.index   # 用于其他 h5ad 的 cell→pos 映射

    # 若 random_scope == "all"，提前计算全部细胞的 pos
    if random_scope == "all":
        all_sc_pool_pos = list(range(len(primary_hdata.obs)))
    else:
        all_sc_pool_pos = None

    # ── 逐周期计算 ──
    all_rows = []
    sig_rows = []

    for cycle in cell_cycles:
        print(f"\n========== {cycle} ==========")

        matrix_accessor = primary_hdata.views_mat[resolution][chrom]

        # single-cell pool（只来自 primary h5ad）
        sc_pool = primary_hdata.obs[
            primary_hdata.obs[label_col] == cycle
        ].index.tolist()

        if len(sc_pool) < 2:
            print(f"  [skip] 单细胞池不足")
            continue

        sc_pool_pos = [primary_hdata.obs.index.get_loc(x) for x in sc_pool]

        # ── 每个算法的 MC SCC ──
        mc_sccs = {}
        group_sizes_primary = None

        for hd, name in zip(hdatas, names):
            mc_ids = hd.metacells[
                hd.metacells[celltype_col] == cycle
            ].index.tolist()

            if len(mc_ids) < 2:
                print(f"  [{name}] metacell 数不足，跳过")
                mc_sccs[name] = []
                continue

            if name == primary_name:
                # 直接从 metacell_data['mat'] 取（已聚合，速度快）
                mc_matrices = [
                    primary_hdata.metacell_data["mat"][resolution][mc_id][chrom]
                    for mc_id in mc_ids
                ]
                # 同时记录分组大小，供 Random 用
                if metacell_col in hd.obs.columns:
                    cyc_obs = hd.obs[hd.obs[label_col] == cycle]
                    group_sizes_primary = [
                        int(max(1, (cyc_obs[metacell_col] == mc_id).sum()))
                        for mc_id in mc_ids
                    ]
                else:
                    fb = max(2, len(sc_pool) // max(2, len(mc_ids)))
                    group_sizes_primary = [fb] * len(mc_ids)
            else:
                # 其他算法：按其 metacell 分配从 primary views_mat 聚合
                mc_matrices = build_metacell_matrices_from_assignment(
                    obs_df=hd.obs,
                    mc_ids=mc_ids,
                    matrix_accessor=matrix_accessor,
                    primary_obs_index=primary_obs_index,
                    cycle=cycle,
                    label_col=label_col,
                    metacell_col=metacell_col,
                    chrom=chrom,
                    desc_prefix=name,
                )

            if len(mc_matrices) < 2:
                print(f"  [{name}] 有效 metacell 矩阵不足，跳过")
                mc_sccs[name] = []
                continue

            scc = calc_pairwise_cor(mc_matrices, max_dist=max_dist,
                                    desc=f"MC ({name} / {cycle})")
            mc_sccs[name] = scc
            for v in scc:
                all_rows.append({"Correlation": v, "Type": name, "CellCycle": cycle})
            print(f"  [{name}] MC mean SCC = {np.mean(scc):.4f}  (n_pairs={len(scc)})")

        # ── Single Cell SCC ──
        n_sc_samples = min(n_sc, len(sc_pool))
        sc_idx = rng.choice(sc_pool, n_sc_samples, replace=False)
        sc_pos = [primary_hdata.obs.index.get_loc(x) for x in sc_idx]
        sc_matrices = [matrix_accessor[p] for p in sc_pos]
        sc_scc = calc_pairwise_cor(sc_matrices, max_dist=max_dist,
                                   desc=f"SC ({cycle})")
        for v in sc_scc:
            all_rows.append({"Correlation": v, "Type": SC_LABEL, "CellCycle": cycle})
        print(f"  [SC] mean SCC = {np.mean(sc_scc):.4f}  (n_pairs={len(sc_scc)})")

        # ── Random SCC（基于 primary 分组大小）──
        if group_sizes_primary is None:
            fb = max(2, len(sc_pool) // 2)
            group_sizes_primary = [fb] * max(2, len(sc_pool) // fb)

        # 根据 random_scope 决定随机池
        rand_pool_pos = all_sc_pool_pos if random_scope == "all" else sc_pool_pos

        rand_distributions = []
        for t in range(n_random):
            rm = build_random_pseudometacells(
                rand_pool_pos, group_sizes_primary, matrix_accessor, rng
            )
            if len(rm) < 2:
                continue
            rand_distributions.append(
                calc_pairwise_cor(rm, max_dist=max_dist,
                                  desc=f"Random ({cycle}) t={t+1}")
            )
        rand_scc = rand_distributions[0] if rand_distributions else []
        for v in rand_scc:
            all_rows.append({"Correlation": v, "Type": RANDOM_LABEL, "CellCycle": cycle})
        rand_mean = np.mean(rand_scc) if rand_scc else np.nan
        print(f"  [Random] mean SCC = {rand_mean:.4f}  (n_pairs={len(rand_scc)})")

        # ── 假设检验：primary vs 其他，primary vs Random，primary vs SC ──
        primary_vals = np.array(mc_sccs.get(primary_name, []))
        comparisons = (
            [(nm, np.array(mc_sccs.get(nm, []))) for nm in names[1:]]
            + [(RANDOM_LABEL, np.array(rand_scc))]
            + [(SC_LABEL, np.array(sc_scc))]
        )
        for cmp_name, cmp_vals in comparisons:
            if len(primary_vals) > 0 and len(cmp_vals) > 0:
                _, p = mannwhitneyu(primary_vals, cmp_vals, alternative="two-sided")
                stars = p_to_stars(p)
            else:
                p, stars = np.nan, "na"
            sig_rows.append({
                "CellCycle": cycle,
                "Comparison": f"{primary_name} vs {cmp_name}",
                "p": p,
                "Stars": stars,
            })

    df = pd.DataFrame(all_rows)
    sig_df = pd.DataFrame(sig_rows)

    print("\n--- 显著性检验结果 ---")
    print(sig_df.to_string(index=False))

    if df.empty:
        print("无数据可绘图，退出。")
        return df, sig_df

    # ── 绘图 ──
    type_order = names + [RANDOM_LABEL, SC_LABEL]
    present = [t for t in type_order if t in df["Type"].unique()]
    palette = _auto_palette(present)

    n_hue = len(present)
    half = 0.4
    offsets = np.linspace(-half * (n_hue - 1) / n_hue,
                          half * (n_hue - 1) / n_hue, n_hue)
    hue_offset = {t: o for t, o in zip(present, offsets)}

    sns.set_theme(
        style="white",
        context="paper",
        rc={"axes.linewidth": 1.2, "font.size": 11,
            "xtick.major.width": 1.0, "ytick.major.width": 1.0},
    )

    fig_w = max(8.0, 2.0 * len(cell_cycles) * n_hue / 3 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=dpi)

    if plot_style == "box":
        sns.boxplot(
            data=df, x="CellCycle", y="Correlation",
            hue="Type", order=cell_cycles, hue_order=present,
            palette=palette, linewidth=1.0, flierprops={"marker": "none"},
            ax=ax,
        )
    else:
        sns.violinplot(
            data=df, x="CellCycle", y="Correlation",
            hue="Type", order=cell_cycles, hue_order=present,
            palette=palette, inner="quartile", cut=0,
            linewidth=1.0, saturation=0.95, density_norm="width",
            ax=ax,
        )

    # 清理图例
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq and l in present:
            uniq[l] = h
    ax.legend(
        [uniq[t] for t in present if t in uniq],
        [t for t in present if t in uniq],
        title="Group", frameon=False,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
    )

    # 显著性标注
    all_y = df["Correlation"].to_numpy()
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_range = max(1e-6, y_max - y_min)
    step = 0.08 * y_range
    bar_h = 0.018 * y_range

    for ci, cycle in enumerate(cell_cycles):
        dcyc = df[df["CellCycle"] == cycle]
        cyc_max = float(np.nanmax(dcyc["Correlation"])) if len(dcyc) else y_max
        base_y = cyc_max + 0.04 * y_range
        x_primary = ci + hue_offset.get(primary_name, 0)

        cyc_sigs = [r for r in sig_rows if r["CellCycle"] == cycle]
        for level, row in enumerate(cyc_sigs):
            cmp_name = row["Comparison"].split(" vs ", 1)[1]
            stars = row["Stars"]
            if stars == "na":
                continue
            x_cmp = ci + hue_offset.get(cmp_name, 0)
            add_sig_bar(ax, x_primary, x_cmp, base_y + level * step, bar_h, stars)

    n_cmp = max(len(names) + 1, 2)
    ax.set_ylim(y_min - 0.05 * y_range,
                y_max + (0.1 + n_cmp * 0.09) * y_range)
    ax.set_title("SCC Comparison Across Cell-Cycle Phases", fontsize=13, pad=10)
    ax.set_xlabel("Cell-Cycle Phase")
    ax.set_ylabel("Pairwise SCC")
    # ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    sns.despine(ax=ax, top=False, right=False)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"\n图已保存：{output}")
    plt.show()

    return df, sig_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="SCC comparison across metacell algorithms."
    )
    p.add_argument("--hdatas", nargs="+", required=True,
                   help="h5ad 路径列表，第一个为你的算法（需含 views_mat）")
    p.add_argument("--names", nargs="+", default=None,
                   help="每个 h5ad 的显示名称（与 --hdatas 等长，可选）")
    p.add_argument("--cell_cycles", nargs="+", default=["G1", "early-S", "late-S-G2"])
    p.add_argument("--chrom", default="chr11")
    p.add_argument("--resolution", type=int, default=1_000_000)
    p.add_argument("--max_dist", type=int, default=50)
    p.add_argument("--n_sc", type=int, default=50,
                   help="每个周期抽取的单细胞样本数上限")
    p.add_argument("--n_random", type=int, default=20,
                   help="随机伪 Metacell 重复次数")
    p.add_argument("--random_scope", default="cycle", choices=["cycle", "all"],
                   help="Random 基线的细胞池：cycle=周期内随机（默认），all=全部细胞随机")
    p.add_argument("--rng_seed", type=int, default=20260422)
    p.add_argument("--output", default="scc_comparison_all.pdf")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--plot_style", default="box", choices=["violin", "box"],
                   help="图形类型：violin=小提琴图（默认），box=箱线图")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        hdata_paths=args.hdatas,
        names=args.names,
        cell_cycles=args.cell_cycles,
        chrom=args.chrom,
        resolution=args.resolution,
        max_dist=args.max_dist,
        n_sc=args.n_sc,
        n_random=args.n_random,
        rng_seed=args.rng_seed,
        output=args.output,
        dpi=args.dpi,
        random_scope=args.random_scope,
        plot_style=args.plot_style,
    )
