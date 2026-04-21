#!/usr/bin/env Rscript
# ============================================================
# Phase 0: Build QC tables (output/tables/)
#
# Generates all QC statistics tables from the imported single-cell
# misha tracks. Must be run before 01_build_pool_track.r.
#
# This step requires the individual cell tracks to already be
# imported into the misha trackdb (schic2_mm9/trackdb/).
# If they are not yet imported, see the import step in apa_loop.R.
#
# Tables generated (into output/tables/):
#   glob_decay.txt
#   glob_decay_res_step*.txt
#   chrom_decay_res_step*_p1_of1.txt
#   tor_stat.txt  (and partitioned tor_stat_p*.txt)
#   chrom_stat.txt
#   chrom_marg.txt
#   chrom_marg_z.txt
#   chrom_marg_enr.txt
#   fend_cis_dup.txt
#   trans_dup_contacts.txt
#   sch_cis_contact_mul.txt
#   sch_trans_contact_mul.txt
#   sch_trans_pairs_contact_count_mul.txt
#   sch_n_distinct_fends_per_cell.txt
#   hyb_2i_es_batch.txt  (must already exist — not generated here)
#
# Usage:
#   cd /path/to/stark
#   Rscript loop_pipeline/00_build_qc_tables.r
#
# Estimated time: 30 min – several hours depending on cluster setup.
# ============================================================

STARK_DIR <- normalizePath(".")   # must be run from stark/
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

# Fix redb path
sch_redb_dir <<- file.path(sch_data_dir, "seq/redb")

# ---- check tracks exist ----
nms <- gtrack.ls(sch_track_base)
if (length(nms) == 0) {
  stop(sprintf(
    "No tracks found matching '%s'. Have you imported the single-cell tracks into misha?",
    sch_track_base))
}
message(sprintf("Found %d single-cell tracks.", length(nms)))

# ---- create output dirs ----
dir.create(sch_table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(sch_fig_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(sch_rdata_dir, recursive = TRUE, showWarnings = FALSE)

# ---- check batch file (must exist already) ----
batch_file <- sprintf("%s/%s", sch_table_dir, sch_batch_fn)
if (!file.exists(batch_file)) {
  stop(sprintf(
    "Batch file not found: %s\nThis file must be copied manually — it is not auto-generated.",
    batch_file))
}

# ---- build QC tables ----
message("Step 1/7: glob decay...")
sch_create_glob_decay()

message("Step 2/7: glob decay (resolution)...")
sch_create_glob_decay_res()

message("Step 3/7: replication timing stat (tor_stat)...")
sch_create_repli_stat()

message("Step 4/7: chrom stat, marg, marg_z, marg_enr...")
sch_create_chrom_stat_marg_and_z()

message("Step 5/7: fend dup stat...")
sch_create_fend_dup_stat()

message("Step 6/7: trans dup contacts...")
count_dup_trans_contacts_across_all_cells()

message("Step 7/7: chrom pairs contacts + unique fends + reads per contact...")
compute_chrom_pairs_total_contacts()
count_unique_fends_per_cell()
compute_reads_per_contact()

# ---- verify ----
required_files <- c(
  "glob_decay.txt",
  "chrom_stat.txt",
  "chrom_marg.txt",
  "chrom_marg_z.txt",
  "chrom_marg_enr.txt",
  "tor_stat.txt",
  "fend_cis_dup.txt"
)

missing <- required_files[!file.exists(file.path(sch_table_dir, required_files))]
if (length(missing) > 0) {
  warning(sprintf("The following expected files are missing: %s", paste(missing, collapse = ", ")))
} else {
  message(sprintf("SUCCESS: all QC tables written to %s", sch_table_dir))
}
