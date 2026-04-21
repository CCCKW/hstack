# ==========================================
# plot_multi.R — 支持多套数据集比较展示
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
# 【重要】在这里添加你要展示的所有数据集路径
# 等号左边是"数据集名称"（会显示在第一列），右边是对应的 CSV 文件夹路径
dataset_dirs <- c(
    "Dataset1" = "/Users/ckw/warehouse/metacell/stark/result_csv",
    "Dataset2" = "/Users/ckw/warehouse/metacell/stark/cycle_reuslt_csv"
    # 如果有更多数据集，可以继续往下加，例如：
    # "Dataset3" = "/Users/ckw/warehouse/metacell/stark/another_csv"
)

# ==========================================
# 2. 批量读取 CSV 并求均值，加入 Dataset 列
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

# 遍历所有数据集并合并
raw_data <- imap_dfr(dataset_dirs, process_dataset)

if (nrow(raw_data) == 0) {
    stop("❌ 没有读取到任何数据，请检查 dataset_dirs 路径配置！")
}

# ==========================================
# 3. Min-Max 归一化
# ==========================================
# 判断哪些列是指标
metrics_cols <- setdiff(colnames(raw_data), c("Dataset", "Method"))

minmax_normalize <- function(x) {
    rng <- range(x, na.rm = TRUE)
    if (rng[1] == rng[2]) {
        return(rep(0.5, length(x)))
    }
    (x - rng[1]) / (rng[2] - rng[1])
}

data <- raw_data %>%
    # funkyheatmap 需要一个无重复的 id 列
    mutate(id = paste(Dataset, Method, sep = "_")) %>%
    mutate(across(all_of(metrics_cols), minmax_normalize)) %>%
    # overall 取各指标归一化后的均值
    mutate(overall = rowMeans(select(., all_of(metrics_cols)), na.rm = TRUE)) %>%
    relocate(id, Dataset, Method) %>%
    # 【排序策略】：先按数据集顺序，每个数据集内按 overall 降序
    mutate(Dataset_factor = factor(Dataset, levels = names(dataset_dirs))) %>%
    arrange(Dataset_factor, desc(overall)) %>%
    select(-Dataset_factor)

# === 可选美化：在每个数据集的中间行显示名称（伪"跨行合并居中"效果） ===
data <- data %>%
    group_by(Dataset) %>%
    mutate(
        mid_row = ceiling(n() / 2),
        Dataset_show = if_else(row_number() == mid_row, Dataset, "")
    ) %>%
    ungroup() %>%
    select(-mid_row)

print("=== 多数据集处理后数据预览 === ")
print(data)

# ==========================================
# 4. 构建列注释信息
# ==========================================
num_metrics <- length(metrics_cols)

column_info <- tibble(
    id = c("Dataset_show", "Method", metrics_cols, "overall"),
    group = c("", "", rep("Indicators", num_metrics), "overall"),
    name = c("Dataset", "Method", metrics_cols, "Overall Score"),
    geom = c("text", "text", rep("funkyrect", num_metrics), "bar"),
    palette = c(NA, NA, rep("palette2", num_metrics), "palette1"),
    options = c(
        list(list(hjust = 0.5, width = 5)), # Dataset 列样式 (水平居中对齐)
        list(list(hjust = 0, width = 5)), # Method 列样式
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
# 5. 绘制 Funky Heatmap
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

# 图表高度自适应：数据越多，图片越高，防止文字挤作一团
plot_height <- max(6, nrow(data) * 0.35 + 2)

ggplot2::ggsave("technology_comparison_heatmap_multi.pdf", plot, width = 12, height = plot_height)
cat("✅ 绘图完成！已保存为 technology_comparison_heatmap_multi.pdf\n")
