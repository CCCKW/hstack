========================================================
loop_pipeline: 生成 loop_paper.txt（完整 paper 流程）
========================================================

目标
----
生成与 paper.r fig3_conv_ctcf_loops_by_cc 完全一致的 loop 文件，
输出到 output/loops/loop_paper.txt。

当前 output/loops/loop.txt 是 fallback 版本（无 contact map 过滤），
不适合用于 APA 分析。

前提条件（已具备）
------------------
- stark/contact_maps/schic_hyb_1C{DES,DU,DX1-4}_adj_files/  — 原始单细胞 contact map
- stark/output/tables/                                        — QC 统计表（glob_decay、tor_stat 等）
- stark/schic2_mm9/trackdb/                                   — misha 基因组数据库
- stark/config/hyb_2i_params.r                                — 参数配置

R 包依赖
--------
  misha, shaman, plyr, dplyr, tidyr, KernSmooth, RColorBrewer, gtools

安装（如未安装）：
  install.packages(c("plyr","dplyr","tidyr","KernSmooth","RColorBrewer","gtools"))
  # misha 和 shaman 为内部包，需从 tanaylab 源安装：
  # install.packages("misha", repos="https://tanaylab.github.io/repo")
  # install.packages("shaman", repos="https://tanaylab.github.io/repo")


运行步骤（务必按顺序）
----------------------

所有步骤均在 stark/ 目录下运行：

  cd /path/to/stark

Step 0a — 导入单细胞 contact tracks 到 misha DB（首次必须）
  Rscript loop_pipeline/00a_import_cell_tracks.r

  作用：将 contact_maps/ 下所有 adj 文件逐个导入为 misha 2D track
        (scell.nextera.1CDES_p1_C3 等，共约数百个)
  前提：schic2_mm9/ 数据库目录和 GATC.fends 文件必须存在
  验证：gtrack.ls("scell.nextera") 应返回数百条 track

Step 0b — 计算 QC 统计表（30分钟~数小时）
  # 先拷贝批次文件：
  cp config/hyb_2i_es_batch.txt output/tables/
  Rscript loop_pipeline/00b_build_qc_tables.r

  作用：计算并写入 output/tables/ 下的所有 QC 文件：
        glob_decay.txt, chrom_stat.txt, tor_stat.txt 等（共约15个）
  前提：Step 0a 已完成（cell tracks 在 misha DB 中）
  跳过条件：output/tables/ 下已有所有文件则可跳过，直接从 Step 1 开始

Step 1 — 构建 pooled contact track（数分钟到1小时）
  Rscript loop_pipeline/01_build_pool_track.r

  作用：将所有 good-QC 单细胞 adj 文件聚合为 misha 2D track
        "scell.nextera.pool_good_hyb_2i_all_es"
  验证：在 R 中执行 gtrack.exists("scell.nextera.pool_good_hyb_2i_all_es") 应返回 TRUE

Step 2 — SHAMAN shuffle + kNN score（数小时，计算密集）
  Rscript loop_pipeline/02_shaman_score.r

  作用：
    a) shuffle —— 对 pooled track 生成随机置换的 null 分布
    b) score  —— 对每个接触点用 kNN 评分，生成
                 "scell.nextera.pool_good_hyb_2i_all_es_score"
  说明：
    - 这是耗时最长的步骤，建议在服务器/高性能机器上运行
    - 如果已有 shuffle/score track，脚本会自动跳过
    - 可通过修改脚本顶部的 RUN_SHUFFLE/RUN_SCORE 变量单独控制
  验证：gtrack.exists("scell.nextera.pool_good_hyb_2i_all_es_score") == TRUE

Step 3 — 生成 loop_paper.txt（数分钟）
  Rscript loop_pipeline/03_generate_loop.r

  作用：以 paper.r 完全相同的参数调用 .get_conv_ctcf_with_tor()，
        同时执行两道过滤：
          1) contact focal enrichment >= 0.8 分位（依赖 pool track）
          2) SHAMAN kNN score >= 60（依赖 score track）
        输出到 output/loops/loop_paper.txt
  验证：wc -l output/loops/loop_paper.txt
        行数应少于 loop.txt（过滤更严格）


过滤参数说明（与 paper.r 完全一致）
--------------------------------------
  ctcf_chip_q         = 0.99  # CTCF ChIP 峰值 quantile 阈值
  tor_horiz           = 2e5   # 复制时序窗口 ±200 kb
  band                = c(2e5, 1e6)  # loop 距离范围 200 kb ~ 1 Mb
  pool_loop_extend    = 1e4   # focal 窗口 ±10 kb
  pool_loop_extend_bg = 3e4   # background 窗口 ±30 kb
  pool_loop_foc_enr_q = 0.8   # 保留 focal/bg 富集值前 20%
  min_d_score         = 60    # SHAMAN kNN 评分阈值
  only_non_overlap    = TRUE  # 去除锚点重叠的冗余 loop


输出文件
--------
  output/loops/loop_paper.txt
    格式（同 loop.txt）：
      chrom1  start1  end1  chrom2  start2  end2
      chr1    ...     ...   chr1    ...     ...


文件说明
--------
  01_build_pool_track.r  — Phase 1，构建 contact map 聚合 track
  02_shaman_score.r      — Phase 2，SHAMAN shuffle + score
  03_generate_loop.r     — Phase 3，调用 .get_conv_ctcf_with_tor 生成 loop
  README.txt             — 本文件
