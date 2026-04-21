#!/usr/bin/env Rscript
# ============================================================
# Phase 1: Build pooled contact track
#
# Aggregates all good-QC single-cell adj files into a single
# misha 2D track (pool_tn = scell.nextera.pool_good_hyb_2i_all_es).
#
# Usage:
#   cd /Users/ckw/warehouse/metacell/stark
#   Rscript loop_pipeline/01_build_pool_track.r
# ============================================================

STARK_DIR <- normalizePath(".")   # run from stark/
CONFIG    <- file.path(STARK_DIR, "config/hyb_2i_params.r")

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
library(gtools)   # for mixedorder

source("map3c/TG3C/params.r")
source("map3c/TG3C/PipeStats.r")
source("map3c/TG3C/analyzeHiC.r")
source("utility_fxns.r")
source("insulation.r")
source("ordering.r")
source("analyzeScHiC.r")

# ---- load config & QC tables ----
message("Loading config and QC tables...")
sch_load_config(CONFIG)

# Fix paths if config still points to a non-existent directory (e.g., original Mac paths)
if (!dir.exists(sch_groot)) {
  candidate <- file.path(STARK_DIR, "schic2_mm9/trackdb")
  if (dir.exists(candidate)) {
    sch_data_dir  <<- file.path(STARK_DIR, "schic2_mm9")
    sch_groot     <<- candidate
    sch_extfiles_dir <<- file.path(STARK_DIR, "schic2_mm9/rawdata")
    gdb.init(sch_groot)
    message(sprintf("Using misha DB: %s", sch_groot))
  } else {
    stop(sprintf("misha DB not found.\n  Tried: %s\n  Tried: %s", sch_groot, candidate))
  }
}
if (!dir.exists(sch_table_dir)) {
  sch_table_dir <<- file.path(STARK_DIR, "output/tables")
  sch_fig_dir   <<- file.path(STARK_DIR, "output/figs")
  sch_rdata_dir <<- file.path(STARK_DIR, "output/rdata")
}

# The redb lives under seq/redb, not directly under schic2_mm9/redb
sch_redb_dir <<- file.path(sch_data_dir, "seq/redb")

sch_glob_decay    <<- read.table(sprintf("%s/glob_decay.txt",    sch_table_dir), header = TRUE)
sch_glob_decay_res <<- read.table(
  sprintf("%s/glob_decay_res_step%s.txt", sch_table_dir, as.character(sch_decay_step)),
  header = TRUE, sep = "\t")
sch_chrom_decay_res <<- read.table(
  sprintf("%s/chrom_decay_res_step%s_p1_of1.txt", sch_table_dir, as.character(sch_decay_step)),
  header = TRUE, sep = "\t")
sch_chrom_stat    <<- read.table(sprintf("%s/chrom_stat.txt",     sch_table_dir), header = TRUE)
sch_chrom_marg    <<- read.table(sprintf("%s/chrom_marg.txt",     sch_table_dir), header = TRUE)
sch_chrom_marg_z  <<- read.table(sprintf("%s/chrom_marg_z.txt",   sch_table_dir), header = TRUE)
sch_chrom_marg_enr <<- read.table(sprintf("%s/chrom_marg_enr.txt", sch_table_dir), header = TRUE)
sch_tor_stat      <<- read.table(sprintf("%s/tor_stat.txt",       sch_table_dir), header = TRUE)
sch_fend_dup      <<- read.table(sprintf("%s/fend_cis_dup.txt",   sch_table_dir), header = TRUE)
sch_batch         <<- read.table(
  sprintf("%s/%s", sch_table_dir, sch_batch_fn),
  header = TRUE, stringsAsFactors = FALSE)

# ---- compute good cells (QC filter only, skip cell-cycle clustering) ----
sch_select_qc_cells()
message(sprintf("Good cells after QC: %d", length(sch_good_cells)))

# ---- check if pool track already exists ----
if (gtrack.exists(pool_tn)) {
  message(sprintf("Pool track '%s' already exists. Skipping creation.", pool_tn))
  message("Delete it with gtrack.rm(pool_tn, force=TRUE) if you want to rebuild.")
  quit(save = "no", status = 0)
}

# ---- build pool track ----
haploid <<- sch_haploid   # required by sch_create_pooled_track_wrapper

message(sprintf("Building pool track '%s' from good cells...", pool_tn))
sch_create_pooled_track_wrapper(sch_good_cells, pool_tn)

if (gtrack.exists(pool_tn)) {
  message(sprintf("SUCCESS: pool track '%s' created.", pool_tn))
} else {
  stop("Pool track creation failed.")
}
