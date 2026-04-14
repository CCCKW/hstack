library(SuperCell)
library(reticulate)

# to current dir
setwd("/Users/ckw/warehouse/metacell/stark")

# 读取npy文件
np <- import("numpy")
pca_mat <- np$load("/Users/ckw/warehouse/metacell/stark/test_output/pca_vec_500000.npy")

# 构建metacell
for (num in 11:40) {
  gamma <- 700 / num
  SC <- SCimplify_from_embedding(
    X = pca_mat,
    k.knn = 5,
    gamma = gamma
  )

  # 输出结果
  result <- data.frame(
    cell_id = 1:nrow(pca_mat),
    metacell_id = SC$membership
  )

  write.csv(result, paste0("./supercell/metacell_membership_", num, ".csv"), row.names = FALSE)
}
