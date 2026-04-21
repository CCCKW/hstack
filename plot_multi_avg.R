# ==========================================
# plot_multi_avg.R — 多数据集平均结果的可视化
# ==========================================

library(dplyr)
library(purrr)
library(readr)
library(stringr)
library(tibble)
library(funkyheatmap)
setwd("/Users/ckw/warehouse/metacell/stark")

# ==========================================
# 1. 定义多套数据集的名称与对应的文件夹路径
# ==========================================
dataset_dirs <- c(
    "Dataset1" = "/Users/ckw/warehouse/metacell/stark/result_csv",
    "Dataset2" = "/Users/ckw/warehouse/metacell/stark/cycle_reuslt_csv"
    # 如果有更多数据集，可以继续添加
)

# ==========================================
# 2. 批量读取 CSV 并合并
# ==========================================
process_dataset <- function(csv_dir, dataset_name) {
    if (!dir.exists(csv_dir)) {
        warning(paste("⚠️ 文件夹不存在，跳过：", csv_dir))
        return(NULL)
    }
    
    file_list <- list.files(path = csv_dir, pattern = "\\.csv$", full.names = TRUE)
    if (length(file_list) == 0) {
        warning(paste("⚠️ 文件夹下没有找到 CSV 文件，跳过：", csv_dir))
        return(NULL)
    }
    
    process_file <- function(filepath) {
        tech_name <- str_remove(basename(filepath), "\\.csv$")
        df <- read_csv(filepath, show_col_types = FALSE)
        df <- df %>% select(-starts_with("..."))
        
        # 对数值列求平均值
        df_mean <- df %>%
            summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
            mutate(Method = tech_name)
        return(df_mean)
    }
    
    dataset_data <- map_dfr(file_list, process_file) %>%
        mutate(Dataset = dataset_name) %>%
        relocate(Dataset, Method)
    
    return(dataset_data)
}

raw_data <- imap_dfr(dataset_dirs, process_dataset)

if (nrow(raw_data) == 0) {
    stop("❌ 没有读取到任何数据，请检查 dataset_dirs 路径配置！")
}

# ==========================================
# 3. 跨数据集计算每种方法的平均值
# ==========================================
# 把同一方法在不同数据集中的指标取算术平均
data_avg <- raw_data %>%
    select(-Dataset) %>%
    group_by(Method) %>%
    summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
    ungroup()

print("=== 各数据集综合平均原始数值 ===")
print(data_avg)

# ==========================================
# 4. Min-Max 归一化并计算 Overall
# ==========================================
metrics_cols <- setdiff(colnames(data_avg), "Method")

minmax_normalize <- function(x) {
    rng <- range(x, na.rm = TRUE)
    if (rng[1] == rng[2]) {
        return(rep(0.5, length(x)))
    }
    (x - rng[1]) / (rng[2] - rng[1])
}

data <- data_avg %>%
    # funkyheatmap 需要一个无重复的 id 列
    mutate(id = Method) %>%
    mutate(across(all_of(metrics_cols), minmax_normalize)) %>%
    # overall 取各指标归一化后的均值
    mutate(overall = rowMeans(select(., all_of(metrics_cols)), na.rm = TRUE)) %>%
    relocate(id) %>%
    # 全局按照 overall 综合得分降序排列
    arrange(desc(overall))

# ==========================================
# 5. 构建列注释信息
# ==========================================
num_metrics <- length(metrics_cols)

column_info <- tibble(
    id = c("id", metrics_cols, "overall"),
    group = c("", rep("Indicators", num_metrics), "overall"),
    name = c("Method", metrics_cols, "Overall Score"),
    geom = c("text", rep("funkyrect", num_metrics), "bar"),
    palette = c(NA, rep("palette2", num_metrics), "palette1"),
    options = c(
        list(list(hjust = 0, width = 6)), # id (Method) 列左对齐
        rep(list(list()), num_metrics),
        list(list(width = 4, legend = FALSE))
    )
)

column_groups <- tibble(
    group = c("Indicators", "overall"),
    palette = c("palette2", "palette1")
)

palettes <- list(
    palette1 = "Blues",
    palette2 = "Blues"
)

# ==========================================
# 6. 绘制 Funky Heatmap
# ==========================================
plot <- suppressWarnings(
    funky_heatmap(
        data = data,
        column_info = column_info,
        column_groups = column_groups,
        palettes = palettes,
        position_args = position_arguments(expand_xmax = 2)
    )
)

print(plot)

# 因为是跨数据集聚合，最终的方法数=独特的Method个数
plot_height <- max(6, nrow(data) * 0.35 + 2)

ggplot2::ggsave("technology_comparison_heatmap_avg.pdf", plot, width = 10, height = plot_height)
cat("✅ 绘图完成！已保存为 technology_comparison_heatmap_avg.pdf\n")
