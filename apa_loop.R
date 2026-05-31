setwd("/Users/ckw/warehouse/metacell/stark")
library(misha)

# define the genomic database we're working on
mm9_db <- "/Users/ckw/warehouse/metacell/stark/schic2_mm9"
gdb.init(sprintf("%s/trackdb", mm9_db))

# directory where contact maps were unpacked (UPDATE to your path)
data_dir <- "/Users/ckw/warehouse/metacell/stark/contact_maps"
base_dir <- data_dir
support_sge <- FALSE

# get the list of directories to work on
dirs = list.files(data_dir)

# parse dirs into track names
nms = paste0("scell.nextera.", gsub("-", "_", gsub(".", "_", dirs, fixed=T), fixed=T))

# fends file is the list of GATC fragment ends
fends = sprintf("%s/seq/redb/GATC.fends", mm9_db)

# create tracks directory 
dir.create(sprintf("%s/trackdb/tracks/scell/nextera", mm9_db), showWarnings=F, recursive=T)

# uploading contact files to misha
if (support_sge) {
    # build the commands
    commands = paste0("gtrack.2d.import_contacts(\"", nms, "\", \"\", \"", paste(base_dir, dirs, "adj", sep="/"), "\", fends, allow.duplicates=F)", collapse=",")
    
    # submit jobs to the sge cluster
    res = eval(parse(text=paste("gcluster.run2(",commands,")")))
}
else {
  # upload cell by cell
  for (i in seq_along(nms)) {
    gtrack.2d.import_contacts(nms[i], "", sprintf("%s/%s/adj", data_dir, dirs[i]), fends, allow.duplicates=F)
  }
}