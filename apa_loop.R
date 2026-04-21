setwd("/Users/ckw/warehouse/metacell/stark")

library(misha)
gdb.init("/Users/ckw/warehouse/metacell/stark/schic2_mm9/trackdb")
fends <- "/Users/ckw/warehouse/metacell/stark/schic2_mm9/seq/redb/GATC.fends"
data_dir <- "/Users/ckw/warehouse/metacell/stark/contact_maps"

batch_dirs <- list.files(data_dir, full.names = T)

for (batch_dir in batch_dirs) {
  cells <- list.files(batch_dir)
  for (cell in cells) {
    nm <- paste0("scell.nextera.", gsub("-", "_", gsub(".", "_", cell, fixed = T), fixed = T))
    adj_path <- sprintf("%s/%s/adj", batch_dir, cell)
    message("importing ", nm)
    gtrack.2d.import_contacts(nm, "", adj_path, fends, allow.duplicates = F)
  }
}

# install.packages("devtools")


source("/Users/ckw/warehouse/metacell/stark/paper.r")

# 在source("paper.r")之后，sch_build_all之前运行这个
0
# 加载所有依赖
library(gtools)
library(plotrix)
library(tglkmeans)
library(misha)
library(shaman)
library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(KernSmooth)
library(RColorBrewer)

TGLKMeans <- TGL_kmeans

gcluster.run <- function(...) {
  calls <- as.list(match.call())[-1]
  res <- vector("list", length(calls))
  for (i in seq_along(calls)) {
    res[[i]] <- list(
      retv = eval(calls[[i]], envir = parent.frame()),
      err = NULL
    )
  }
  res
}

source("paper.r")

sch_batch <<- read.table("/Users/ckw/warehouse/metacell/stark/config/hyb_2i_es_batch.txt",
  header = T, stringsAsFactors = F
)

sch_build_all(spec_params_fn = "/Users/ckw/warehouse/metacell/stark/config/hyb_2i_params.r")
