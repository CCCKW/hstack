#!/usr/bin/env Rscript
# ============================================================
# Phase 2: SHAMAN shuffle + kNN score track
#
# Generates the null distribution (shuffle) and computes the
# SHAMAN kNN enrichment score for the pooled contact track.
# This is the most computationally intensive step.
#
# Estimated time: several hours on a workstation.
# For faster execution, run on an HPC cluster (SGE/SLURM).
#
# Usage:
#   cd /Users/ckw/warehouse/metacell/stark
#   Rscript loop_pipeline/02_shaman_score.r
#
# Optional: run shuffle and score separately by setting
#   RUN_SHUFFLE=TRUE/FALSE and RUN_SCORE=TRUE/FALSE below.
# ============================================================

STARK_DIR   <- normalizePath(".")   # run from stark/
CONFIG      <- file.path(STARK_DIR, "config/hyb_2i_params.r")
RUN_SHUFFLE <- TRUE   # set FALSE to skip if shuffle already done
RUN_SCORE   <- TRUE   # set FALSE to skip if score track already exists

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(STARK_DIR)

# ---- dependencies ----
library(misha)
library(shaman)
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
if (!dir.exists(sch_rdata_dir)) {
  sch_rdata_dir <<- file.path(STARK_DIR, "output/rdata")
}

# ---- check prerequisites ----
if (!gtrack.exists(pool_tn)) {
  stop(sprintf(
    "Pool track '%s' not found. Run 01_build_pool_track.r first.", pool_tn))
}

dir.create(sch_rdata_dir, recursive = TRUE, showWarnings = FALSE)

score_tn <- paste0(pool_tn, "_score")

# ---- SHAMAN: generate shuffled (null) track ----
if (RUN_SHUFFLE) {
  shuffle_tn <- paste0(pool_tn, "_shuffle")
  if (gtrack.exists(shuffle_tn)) {
    message(sprintf("Shuffle track '%s' already exists. Skipping.", shuffle_tn))
  } else {
    message("Running SHAMAN shuffle (generates null distribution)...")
    message("This step can take several hours.")
    # Disable SGE for local execution
    options(shaman.sge_support = 0)
    shaman_shuffle_hic_track(
      track_db    = sch_groot,
      obs_track_nm = pool_tn,
      work_dir    = sprintf("%s/", sch_rdata_dir)
    )
    message("Shuffle complete.")
  }
} else {
  message("Skipping shuffle (RUN_SHUFFLE=FALSE).")
}

# ---- SHAMAN: compute kNN score track ----
if (RUN_SCORE) {
  if (gtrack.exists(score_tn)) {
    message(sprintf("Score track '%s' already exists. Skipping.", score_tn))
  } else {
    message(sprintf("Computing SHAMAN kNN score -> '%s'...", score_tn))
    message("This step can take several hours.")
    options(shaman.sge_support = 0)
    shaman_score_hic_track(
      track_db      = sch_groot,
      obs_track_nms = pool_tn,
      score_track_nm = score_tn
    )
    message("Score track complete.")
  }
} else {
  message("Skipping score (RUN_SCORE=FALSE).")
}

# ---- verify ----
if (gtrack.exists(score_tn)) {
  message(sprintf("SUCCESS: score track '%s' is ready.", score_tn))
} else {
  warning(sprintf(
    "Score track '%s' not found. Check errors above.", score_tn))
}
