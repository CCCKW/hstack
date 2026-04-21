# ==========================================
# 1. 安装并加载必要的 R 包
# ==========================================
# 如果你没有安装这些包，请取消注释下面这行代码进行安装：
# install.packages(c("funkyheatmap", "dplyr", "purrr", "readr", "stringr", "tibble"))

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
# 【重要】请将这里的路径替换为你存放 CSV 文件的实际文件夹路径
# 例如：csv_dir <- "C:/Users/Data/Technologies"
csv_dir <- "/Users/ckw/warehouse/metacell/stark/result_csv"

# 获取该目录下所有的 .csv 文件路径
file_list <- list.files(path = csv_dir, pattern = "\\.csv$", full.names = TRUE)

if (length(file_list) == 0) {
    stop("指定的路径下没有找到 CSV 文件，请检查路径！")
}

# 定义一个处理单个文件的函数
process_file <- function(filepath) {
    # 从文件名中提取技术名称 (例如从 "技术A.csv" 提取 "技术A")
    tech_name <- str_remove(basename(filepath), "\\.csv$")

    # 读取 csv 文件
    df <- read_csv(filepath, show_col_types = FALSE)

    # ================= 新增：清理多余列 =================
    # 删除原始表格中无名的行号或索引列 (read_csv会自动命名为 ...1, ...2 等)
    df <- df %>% select(-starts_with("..."))

    # 对所有数值列求平均值 (忽略 NA 值)
    df_mean <- df %>%
        summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
        mutate(id = tech_name)

    return(df_mean)
}

# 批量处理所有文件，并合并成一个数据框（行 = 技术，列 = id + 指标）
data <- map_dfr(file_list, process_file) %>%
    # ================= 新增：计算 overall 并排序 =================
    # 计算每一行的平均值作为 overall 综合得分
    # (注意：如果您的各个指标量纲/单位差异极大，建议先做标准化处理)
    mutate(overall = rowMeans(select(., -id), na.rm = TRUE)) %>%
    relocate(id) %>% # 将 id 移动到最前面，overall 保持在最后
    arrange(desc(overall)) # 根据 overall 降序排列，生成排名

# 打印合并后的数据预览，检查是否正确
print("=== 聚合排序后的数据预览 ===")
print(data)

# ==========================================
# 3. 构建列注释信息 (column_info) 和其他配置
# ==========================================
# 获取所有的具体指标列名（排除 id 和 overall 列）
metrics <- setdiff(colnames(data), c("id", "overall"))
num_metrics <- length(metrics)

# 自动生成 column_info
# - id 列设为 "text" 文本
# - overall 列设为 "bar" 柱状图
# - 其他指标列设为 "funkyrect" (带颜色的圆角矩形，大小代表数值)
column_info <- tibble(
    id = c("id", metrics, "overall"),
    group = c("", rep("Indicators", num_metrics), "overall"), # 其他指标一组，overall 单独一组
    name = c("", metrics, "Overall Score"), # 图上显示的列名
    geom = c("text", rep("funkyrect", num_metrics), "bar"), # 指标用 funkyrect，overall 用 bar
    palette = c(NA, rep("palette2", num_metrics), "palette1"), # 分别使用不同的调色板
    options = c(
        list(list(hjust = 0, width = 6)), # id 列的选项（左对齐，宽度6）
        rep(list(list()), num_metrics), # 其他列使用默认选项
        list(list(width = 4, legend = FALSE)) # overall 列的选项（宽度4，不显示图例）
    )
)

# 显式定义列分组信息
column_groups <- tibble(
    group = c("Indicators", "overall"),
    palette = c("palette2", "palette1")
)

# 显式定义调色板
palettes <- list(
    palette1 = "Blues", # overall 排名使用灰色柱子
    palette2 = "Blues" # 具体指标使用蓝色块
)

# 打印列注释信息预览
print("=== 列注释信息 column_info 预览 ===")
print(column_info)


# ==========================================
# 4. 绘制 Funky Heatmap
# ==========================================
# 生成热图 (使用 suppressWarnings 屏蔽底层包内部的良性警告)
plot <- suppressWarnings(
    funky_heatmap(
        data = data,
        column_info = column_info,
        column_groups = column_groups,
        palettes = palettes,
        # position_args 可以用来调整图表的边距和大小
        position_args = position_arguments(expand_xmax = 2)
    )
)

# 显示图表
print(plot)

# 如果需要保存图片，可以使用 ggsave (funkyheatmap 底层是 ggplot2 / patchwork)
ggplot2::ggsave("technology_comparison_heatmap.pdf", plot, width = 10, height = 6)
# ggplot2::ggsave("technology_comparison_heatmap.png", plot, width = 10, height = 6, bg = "white")
