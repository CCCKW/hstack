#!/usr/bin/env Rscript
# ============================================================
# Phase 0b: Build QC tables
#
# Computes glob_decay, chrom_stat, tor_stat etc. from the
# individual cell misha tracks. Writes results to output/tables/.
#
# Must be run AFTER 00a_import_cell_tracks.r.
#
# Usage:
#   cd /path/to/stark
#   Rscript loop_pipeline/00b_build_qc_tables.r
# ============================================================

STARK_DIR <- normalizePath(".")
CONFIG    <- file.path(STARK_DIR, "config/hyb_2i_params.r")

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(STARK_DIR)

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

message("Loading config...")
sch_load_config(CONFIG)

# Fix paths if config still points to a non-existent directory (e.g., original Mac paths)
if (!dir.exists(sch_groot)) {
  candidate <- file.path(STARK_DIR, "schic2_mm9/trackdb")
  if (dir.exists(candidate)) {
    sch_data_dir  <<- file.path(STARK_DIR, "schic2_mm9")
    sch_groot     <<- candidate
    sch_redb_dir  <<- file.path(STARK_DIR, "schic2_mm9/seq/redb")
    sch_extfiles_dir <<- file.path(STARK_DIR, "schic2_mm9/rawdata")
    gdb.init(sch_groot)
    message(sprintf("Using misha DB: %s", sch_groot))
  } else {
    stop(sprintf("misha DB not found.\n  Tried: %s\n  Tried: %s\n  Ensure schic2_mm9/ is present under stark/", sch_groot, candidate))
  }
}
if (!dir.exists(sch_table_dir)) {
  sch_table_dir <<- file.path(STARK_DIR, "output/tables")
  sch_fig_dir   <<- file.path(STARK_DIR, "output/figs")
  sch_rdata_dir <<- file.path(STARK_DIR, "output/rdata")
}

# verify cell tracks exist — check each pattern in sch_track_base separately
# (gtrack.ls does not support | alternation in all misha versions)
patterns <- strsplit(sch_track_base, "\\|")[[1]]
all_tracks <- unlist(lapply(patterns, function(p) gtrack.ls(p)))
all_tracks <- unique(all_tracks)
if (length(all_tracks) == 0) {
  stop(sprintf(
    "No tracks matching '%s' found in misha DB.\nRun 00a_import_cell_tracks.r first.",
    sch_track_base
  ))
}
message(sprintf("Found %d cell tracks in DB.", length(all_tracks)))

# create output dirs
dir.create(sch_table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(sch_fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(sch_rdata_dir, recursive = TRUE, showWarnings = FALSE)

# ---- build each QC table ----
# Use gcluster fallback (no SGE)
gcluster.run <- function(...) {
  calls <- as.list(match.call())[-1]
  lapply(calls, function(x) list(retv = eval(x, envir = parent.frame(2)), err = NULL))
}

message("Building glob_decay table...")
sch_create_glob_decay()

message("Building glob_decay_res table...")
sch_create_glob_decay_res()

message("Building replication timing stat...")
sch_create_repli_stat()

message("Building chrom stat / marg / z tables...")
sch_create_chrom_stat_marg_and_z()

message("Building fend dup stat...")
sch_create_fend_dup_stat()

message("Counting duplicate trans contacts...")
count_dup_trans_contacts_across_all_cells()

message("Computing chrom pairs total contacts...")
compute_chrom_pairs_total_contacts()

message("Counting unique fends per cell...")
count_unique_fends_per_cell()

message("Computing reads per contact...")
compute_reads_per_contact()

message(sprintf("All QC tables written to %s", sch_table_dir))
message("Now copy config/hyb_2i_es_batch.txt into output/tables/ if not already there.")
