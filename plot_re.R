# ==========================================
# plot_re.R — 使用 min-max 归一化替代排名归一化
# 原始值差距更直观地体现在图中
# ==========================================

library(dplyr)
library(purrr)
library(readr)
library(stringr)
library(tibble)
library(funkyheatmap)
setwd("/Users/ckw/warehouse/metacell/stark")

# ==========================================
# 2. 批量读取 CSV 并求均值
# ==========================================
csv_dir <- "/Users/ckw/warehouse/metacell/stark/cycle_reuslt_csv"

file_list <- list.files(path = csv_dir, pattern = "\\.csv$", full.names = TRUE)

if (length(file_list) == 0) {
    stop("指定的路径下没有找到 CSV 文件，请检查路径！")
}

process_file <- function(filepath) {
    tech_name <- str_remove(basename(filepath), "\\.csv$")
    df <- read_csv(filepath, show_col_types = FALSE)
    df <- df %>% select(-starts_with("..."))
    df_mean <- df %>%
        summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
        mutate(id = tech_name)
    return(df_mean)
}

raw_data <- map_dfr(file_list, process_file) %>%
    relocate(id)

# ==========================================
# 3. Min-Max 归一化（保留原始数值差距）
# ==========================================
# 对每个指标列做 min-max，让值域拉伸到 [0, 1]
# 这样不同方法之间的真实差距会在热图中清晰可见
metrics_cols <- setdiff(colnames(raw_data), "id")

minmax_normalize <- function(x) {
    rng <- range(x, na.rm = TRUE)
    if (rng[1] == rng[2]) {
        return(rep(0.5, length(x)))
    } # 全部相同时返回0.5
    (x - rng[1]) / (rng[2] - rng[1])
}

data <- raw_data %>%
    mutate(across(all_of(metrics_cols), minmax_normalize)) %>%
    # overall 取各指标归一化后的均值
    mutate(overall = rowMeans(select(., all_of(metrics_cols)), na.rm = TRUE)) %>%
    arrange(desc(overall))

print("=== Min-Max 归一化后数据预览 ===")
print(data)

# ==========================================
# 4. 构建列注释信息
# ==========================================
metrics <- setdiff(colnames(data), c("id", "overall"))
num_metrics <- length(metrics)

column_info <- tibble(
    id = c("id", metrics, "overall"),
    group = c("", rep("Indicators", num_metrics), "overall"),
    name = c("", metrics, "Overall Score"),
    geom = c("text", rep("funkyrect", num_metrics), "bar"),
    palette = c(NA, rep("palette2", num_metrics), "palette1"),
    options = c(
        list(list(hjust = 0, width = 6)),
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

ggplot2::ggsave("technology_comparison_heatmap_minmax.pdf", plot, width = 10, height = 6)
