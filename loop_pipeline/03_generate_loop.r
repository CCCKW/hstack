#!/usr/bin/env Rscript
# ============================================================
# Phase 3: Generate loop_paper.txt
#
# Uses the same parameters as paper.r fig3_conv_ctcf_loops_by_cc:
#   ctcf_chip_q      = 0.99
#   tor_horiz        = 2e5
#   band             = c(2e5, 1e6)
#   pool_loop_extend = 1e4    (focal window ±10 kb)
#   pool_loop_extend_bg = 3e4 (background window ±30 kb)
#   pool_loop_foc_enr_q = 0.8 (keep top-20% by focal enrichment)
#   min_d_score      = 60     (SHAMAN kNN score threshold)
#
# Prerequisites: phases 01 and 02 must have completed successfully.
#
# Usage:
#   cd /Users/ckw/warehouse/metacell/stark
#   Rscript loop_pipeline/03_generate_loop.r
# ============================================================

STARK_DIR <- normalizePath(".")   # run from stark/
CONFIG    <- file.path(STARK_DIR, "config/hyb_2i_params.r")
OUT_FILE  <- file.path(STARK_DIR, "output/loops/loop_paper.txt")

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(STARK_DIR)

# ---- dependencies ----
library(misha)
library(plyr)
library(dplyr)
library(tidyr)
library(KernSmooth)
library(RColorBrewer)
library(gtools)

source("map3c/TG3C/params.r")
source("map3c/TG3C/PipeStats.r")
source("map3c/TG3C/analyzeHiC.r")
source("utility_fxns.r")
source("insulation.r")
source("ordering.r")
source("analyzeScHiC.r")

# ---- load config ----
message("Loading config...")
sch_load_config(CONFIG)

# Fix paths if config still points to a non-existent directory (e.g., original Mac paths)
if (!dir.exists(sch_groot)) {
  candidate <- file.path(STARK_DIR, "schic2_mm9/trackdb")
  if (dir.exists(candidate)) {
    sch_data_dir  <<- file.path(STARK_DIR, "schic2_mm9")
    sch_groot     <<- candidate
    gdb.init(sch_groot)
    message(sprintf("Using misha DB: %s", sch_groot))
  } else {
    stop(sprintf("misha DB not found.\n  Tried: %s\n  Tried: %s", sch_groot, candidate))
  }
}

score_tn <- paste0(pool_tn, "_score")

# ---- check prerequisites ----
if (!gtrack.exists(pool_tn)) {
  stop(sprintf("Pool track '%s' not found. Run 01_build_pool_track.r first.", pool_tn))
}
if (!gtrack.exists(score_tn)) {
  stop(sprintf("Score track '%s' not found. Run 02_shaman_score.r first.", score_tn))
}

# ---- generate loops (paper parameters) ----
message("Generating convergent CTCF loops with paper-style filtering...")
message(sprintf("  pool track : %s", pool_tn))
message(sprintf("  score track: %s", score_tn))

loops <- .get_conv_ctcf_with_tor(
  ctcf_chip_q         = 0.99,
  tor_horiz           = 2e5,
  band                = c(2e5, 1e6),
  pool_q_cutoff       = 0,
  pool_loop_extend    = 1e4,
  pool_loop_extend_bg = 3e4,
  pool_loop_foc_enr_q = 0.8,
  min_d_score         = 60,
  score_tn            = score_tn,
  only_non_overlap_loops = TRUE
)

if (nrow(loops) == 0) {
  stop("No loops returned. Check track data and parameters.")
}

message(sprintf("Loops after all filters: %d", nrow(loops)))

# ---- write output (6-column, same format as loop.txt) ----
keep <- c("chrom1", "start1", "end1", "chrom2", "start2", "end2")
out  <- loops[, intersect(keep, colnames(loops)), drop = FALSE]

dir.create(dirname(OUT_FILE), recursive = TRUE, showWarnings = FALSE)
write.table(out, file = OUT_FILE, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)

message(sprintf("Done: %d loops written to %s", nrow(out), OUT_FILE))
