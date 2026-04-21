#!/usr/bin/env Rscript
# ============================================================
# Phase 0a: Import single-cell contact tracks into misha DB
#
# Reads adj files from contact_maps/ and imports each cell as
# a 2D misha track (scell.nextera.XXX). This must be done
# before any QC table generation.
#
# Usage:
#   cd /path/to/stark
#   Rscript loop_pipeline/00a_import_cell_tracks.r
# ============================================================

STARK_DIR   <- normalizePath(".")   # run from stark/
CONFIG      <- file.path(STARK_DIR, "config/hyb_2i_params.r")
DATA_DIR    <- file.path(STARK_DIR, "contact_maps")   # contains schic_hyb_* subdirs

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(STARK_DIR)

library(misha)
library(gtools)   # mixedorder

# load config (initialises gdb)
source("map3c/TG3C/params.r")
source("map3c/TG3C/PipeStats.r")
source("map3c/TG3C/analyzeHiC.r")
source("utility_fxns.r")
source("insulation.r")
source("ordering.r")
source("analyzeScHiC.r")

sch_load_config(CONFIG)

# Fix paths if config still points to a non-existent directory (e.g., original Mac paths)
if (!dir.exists(sch_groot)) {
  candidate <- file.path(STARK_DIR, "schic2_mm9/trackdb")
  if (dir.exists(candidate)) {
    sch_data_dir  <<- file.path(STARK_DIR, "schic2_mm9")
    sch_groot     <<- candidate
    sch_redb_dir  <<- file.path(STARK_DIR, "schic2_mm9/seq/redb")
    gdb.init(sch_groot)
    message(sprintf("Using misha DB: %s", sch_groot))
  } else {
    stop(sprintf("misha DB not found.\n  Tried: %s\n  Tried: %s", sch_groot, candidate))
  }
}

fends <- file.path(sch_redb_dir, "GATC.fends")
if (!file.exists(fends)) {
  # also try seq/redb
  fends2 <- file.path(STARK_DIR, "schic2_mm9/seq/redb/GATC.fends")
  if (file.exists(fends2)) {
    fends <- fends2
  } else {
    stop(sprintf("FENDS file not found:\n  %s\n  %s\nCheck that schic2_mm9/ is present.", fends, fends2))
  }
}
message(sprintf("Using FENDS: %s", fends))

# discover all batch dirs
batch_dirs <- list.dirs(DATA_DIR, recursive = FALSE, full.names = TRUE)
if (length(batch_dirs) == 0) {
  stop(sprintf("No batch directories found under %s", DATA_DIR))
}

imported <- 0
skipped  <- 0
failed   <- 0

for (batch_dir in sort(batch_dirs)) {
  cells <- list.dirs(batch_dir, recursive = FALSE, full.names = FALSE)
  for (cell in mixedorder(cells)) {
    # convert cell dir name to misha track name
    # e.g. "1CDES_p1.C3" -> "scell.nextera.1CDES_p1_C3"
    nm <- paste0("scell.nextera.", gsub("\\.", "_", cell))

    adj_path <- file.path(batch_dir, cell, "adj")
    if (!file.exists(adj_path)) {
      message(sprintf("  SKIP (no adj): %s", adj_path))
      skipped <- skipped + 1
      next
    }

    # check both misha registry and disk directory (handles partial previous imports)
    track_dir <- file.path(sch_groot, "tracks", gsub("\\.", "/", nm))
    if (gtrack.exists(nm) || dir.exists(track_dir)) {
      skipped <- skipped + 1
      next
    }

    tryCatch({
      gtrack.2d.import_contacts(
        track      = nm,
        description = "",
        contacts   = adj_path,
        fends      = fends,
        allow.duplicates = FALSE
      )
      imported <- imported + 1
      if (imported %% 50 == 0) {
        message(sprintf("  ... imported %d tracks so far", imported))
      }
    }, error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("already exists", msg, ignore.case = TRUE)) {
        skipped <<- skipped + 1
      } else {
        message(sprintf("  ERROR importing %s: %s", nm, msg))
        failed <<- failed + 1
      }
    })
  }
}

message(sprintf(
  "Done. Imported: %d  |  Skipped (exists/no adj): %d  |  Failed: %d",
  imported, skipped, failed
))
if (failed > 0) {
  message("Check errors above. Failed tracks may need manual import.")
}
