#!/usr/bin/env Rscript

# Minimal export: generate one loop txt for downstream APA.
# Usage:
#   Rscript stark/get_loop_txt.r config/hyb_2i_params.r stark/loop.txt

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript stark/get_loop_txt.r <config.r> <out_loop_txt>")
}

orig_wd <- getwd()
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (length(script_file) == 0) {
  script_file <- file.path(orig_wd, "stark/get_loop_txt.r")
}
script_dir <- normalizePath(dirname(script_file), mustWork = TRUE)

config_path <- normalizePath(args[[1]], mustWork = TRUE)
out_file <- args[[2]]
if (!grepl("^/", out_file)) {
  out_file <- file.path(orig_wd, out_file)
}

setwd(script_dir)

source("analyzeScHiC.r")

# Load config to initialize misha DB and track names.
sch_load_config(config_path)

# Prefer the exact paper-style filtering when pooled score track exists.
score_tn <- paste0(pool_tn, "_score")
have_pool_track <- gtrack.exists(pool_tn)
have_score_track <- gtrack.exists(score_tn)

if (have_pool_track && have_score_track) {
  message("Using paper-style loop filtering with pooled score track.")
  loops <- .get_conv_ctcf_with_tor(
    ctcf_chip_q = 0.99,
    tor_horiz = 2e5,
    band = c(2e5, 1e6),
    pool_q_cutoff = 0,
    pool_loop_extend = 1e4,
    pool_loop_extend_bg = 3e4,
    pool_loop_foc_enr_q = 0.8,
    min_d_score = 60,
    score_tn = score_tn,
    only_non_overlap_loops = TRUE
  )
} else {
  message("Pooled track or score track not found; fallback to CTCF+TOR loops (no pool-score filtering).")
  loops <- .get_conv_ctcf_with_tor(
    ctcf_chip_q = 0.99,
    tor_horiz = 2e5,
    band = c(2e5, 1e6),
    pool_q_cutoff = 0,
    pool_loop_extend = 0,
    pool_loop_extend_bg = 0,
    pool_loop_foc_enr_q = 0.8,
    min_d_score = NA,
    score_tn = score_tn,
    only_non_overlap_loops = TRUE
  )
}

if (!nrow(loops)) {
  stop("No loops returned. Try relaxing filters.")
}

keep <- c("chrom1", "start1", "end1", "chrom2", "start2", "end2")
loops_txt <- loops[, keep, drop = FALSE]

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
write.table(loops_txt, file = out_file, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)

message(sprintf("Done: %d loops -> %s", nrow(loops_txt), out_file))
