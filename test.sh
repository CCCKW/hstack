#!/bin/bash

###############################################################################
# WGBS Pipeline - 从FASTQ到BAM的标准流程
# 作者：自动生成 (优化修复版)
# 日期：2026-04-11
# 优化：修复并发失控问题，优化 Bismark 线程配比（适配96核）
###############################################################################

set -o pipefail  # 管道命令任何一步失败都返回错误

export LD_LIBRARY_PATH="/root/ckw/mHapTools-master/htslib-1.10.2/lib:${LD_LIBRARY_PATH}"
export TMPDIR="/home/ssd/tmp"
mkdir -p "${TMPDIR}"

# ============================= 配置参数 =====================================

# 测试模式：设置为1时只处理1个样本后退出，设置为0时处理所有样本
TEST_MODE=0  # 改为0可处理所有样本
TEST_SAMPLES_LIMIT=1  # 测试模式下处理的样本数量

# mHapTools开关：设置为1时执行BAM到mHap的转换
ENABLE_MHAPTOOLS=1  # 改为0可跳过mHapTools转换

# 目录配置
RAW_DATA_DIR="/home/ssd/test_WGBS"
OUTPUT_DIR="/home/ssd/test_output"
LOG_DIR="${OUTPUT_DIR}/logs"

# 参考基因组配置（请填写您的路径）
GENOME_DIR="/root/ckw/hg38"  
GENOME_FASTA="${GENOME_DIR}/hg38.fa"  
BISMARK_INDEX_DIR="${GENOME_DIR}/Bismark_Index"  

# mHapTools配置
CPG_FILE="/root/ckw/hg38/hg38_CpG.gz"  

# 资源配置 (针对 96 核优化)
TOTAL_THREADS=96        
PARALLEL_SAMPLES=4      # 同时处理4个样本
# Bismark: parallel 4 * -p 4 = 16 核心/样本. 4*16 = 64核心，留余量给I/O和其他步骤
THREADS_PER_SAMPLE=20   # Samtools等使用的线程

# 工具参数
TRIM_QUALITY=20  
MIN_LENGTH=20    

# 日志文件
MASTER_LOG="${LOG_DIR}/master.log"
SUCCESS_LOG="${LOG_DIR}/success_samples.txt"
FAILED_LOG="${LOG_DIR}/failed_samples.txt"
PROGRESS_FILE="${LOG_DIR}/progress.txt"

# ============================= 函数定义 =====================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "${MASTER_LOG}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "${MASTER_LOG}"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "${MASTER_LOG}"
}

init_directories() {
    log_info "初始化输出目录..."
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${LOG_DIR}/batch_logs"
    
    touch "${MASTER_LOG}"
    touch "${SUCCESS_LOG}"
    touch "${FAILED_LOG}"
    touch "${PROGRESS_FILE}"
}

check_tools() {
    log_info "检查所需工具..."
    
    local tools=("fastqc" "trim_galore" "bismark" "bismark_genome_preparation" "deduplicate_bismark" "samtools" "bowtie2")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ $ENABLE_MHAPTOOLS -eq 1 ]; then
        if ! command -v "mhaptools" &> /dev/null; then
            missing_tools+=("mhaptools")
        fi
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "缺少以下工具: ${missing_tools[*]}"
        exit 1
    fi
}

prepare_bismark_index() {
    log_info "检查Bismark索引..."
    
    if [ ! -f "${GENOME_FASTA}" ]; then
        log_error "参考基因组文件不存在: ${GENOME_FASTA}"
        exit 1
    fi
    
    if [ ! -d "${BISMARK_INDEX_DIR}/Bisulfite_Genome" ]; then
        log_info "Bismark索引不存在，开始构建索引..."
        mkdir -p "${BISMARK_INDEX_DIR}"
        
        if [ ! -f "${BISMARK_INDEX_DIR}/$(basename ${GENOME_FASTA})" ]; then
            ln -s "${GENOME_FASTA}" "${BISMARK_INDEX_DIR}/"
        fi
        
        bismark_genome_preparation --bowtie2 --parallel 4 "${BISMARK_INDEX_DIR}" 2>&1 | tee -a "${MASTER_LOG}"
        
        if [ $? -eq 0 ]; then
            log_success "Bismark索引构建完成"
        else
            log_error "Bismark索引构建失败"
            exit 1
        fi
    else
        log_info "Bismark索引已存在，跳过构建"
    fi
}

is_sample_processed() {
    local sample_name=$1
    grep -q "^${sample_name}$" "${PROGRESS_FILE}" 2>/dev/null
    return $?
}

mark_sample_done() {
    local sample_name=$1
    echo "${sample_name}" >> "${PROGRESS_FILE}"
}

cleanup_redundant_files() {
    local output_batch_dir=$1
    local sample_basename=$2
    local sample_log=$3
    
    cd "${output_batch_dir}"
    rm -f *_trimming_report.txt
    rm -f *_bismark_bt2_PE_report.txt
    rm -f *.deduplication_report.txt
    rm -f *_fastqc.zip
    rm -f *.flagstat.txt
    rm -rf FastQC
}

process_sample() {
    local batch_dir=$1
    local r1_file=$2
    local r2_file=$3
    local output_batch_dir=$4
    local sample_log=$5
    
    local sample_basename=$(basename "$r1_file" "_R1_001.fastq.gz")
    local sample_name="${batch_dir##*/}/${sample_basename}"
    
    local start_time=$(date +%s)
    log_info "开始处理样本: ${sample_name}" | tee -a "${sample_log}"
    
    if is_sample_processed "${sample_name}"; then
        log_info "样本已处理，跳过: ${sample_name}" | tee -a "${sample_log}"
        return 0
    fi
    
    local work_dir="${OUTPUT_DIR}/tmp/temp_${sample_basename}"
    mkdir -p "${work_dir}"
    
    log_info "[${sample_basename}] Step 1 & 2: Trim Galore去接头和质控..." | tee -a "${sample_log}"
    # 稍微增加核心数，利用系统资源
    trim_galore --paired --fastqc \
        --quality "${TRIM_QUALITY}" \
        --length "${MIN_LENGTH}" \
        --cores 6 \
        --output_dir "${work_dir}" \
        "${r1_file}" "${r2_file}" >> "${sample_log}" 2>&1
    
    if [ $? -ne 0 ]; then
        log_error "[${sample_basename}] Trim Galore失败" | tee -a "${sample_log}"
        return 1
    fi
    
    local trimmed_r1="${work_dir}/$(basename ${r1_file} .fastq.gz)_val_1.fq.gz"
    local trimmed_r2="${work_dir}/$(basename ${r2_file} .fastq.gz)_val_2.fq.gz"
    
    log_info "[${sample_basename}] Step 3: Bismark比对..." | tee -a "${sample_log}"
    # 优化点: --parallel 4 (开启4个bowtie2实例) 和 -p 4 (每个实例4线程)，这样每个样本消耗约 16-20 核，稳定且快
    bismark --bowtie2 \
        --genome "${BISMARK_INDEX_DIR}" \
        --parallel 4 \
        -p 4 \
        --output_dir "${work_dir}" \
        --temp_dir "${work_dir}/temp" \
        -1 "${trimmed_r1}" \
        -2 "${trimmed_r2}" >> "${sample_log}" 2>&1
    
    if [ $? -ne 0 ]; then
        log_error "[${sample_basename}] Bismark比对失败" | tee -a "${sample_log}"
        return 1
    fi
    
    local bismark_bam=$(find "${work_dir}" -name "*_val_1_bismark_bt2_pe.bam" | head -n 1)
    
    log_info "[${sample_basename}] Step 4: 去重复..." | tee -a "${sample_log}"
    deduplicate_bismark --paired --bam \
        --output_dir "${work_dir}" \
        "${bismark_bam}" >> "${sample_log}" 2>&1
    
    local dedup_bam=$(find "${work_dir}" -name "*.deduplicated.bam" | head -n 1)
    
    log_info "[${sample_basename}] Step 5: BAM排序..." | tee -a "${sample_log}"
    local final_bam="${output_batch_dir}/${sample_basename}.sorted.bam"
    # 分配 4 个排序线程，每个线程 4G 内存（总计 16G/样本）
    samtools sort -@ 4 \
        -m 4G \
        -o "${final_bam}" \
        "${dedup_bam}" >> "${sample_log}" 2>&1
    
    log_info "[${sample_basename}] Step 6: 建立索引..." | tee -a "${sample_log}"
    samtools index -@ ${THREADS_PER_SAMPLE} "${final_bam}" >> "${sample_log}" 2>&1
    samtools flagstat -@ ${THREADS_PER_SAMPLE} "${final_bam}" >> "${sample_log}" 2>&1
    
    if [ $ENABLE_MHAPTOOLS -eq 1 ]; then
        log_info "[${sample_basename}] Step 7: BAM转换为mHap格式..." | tee -a "${sample_log}"
        local mhap_file="${output_batch_dir}/${sample_basename}.mhap.gz"
        mhaptools convert -i "${final_bam}" -c "${CPG_FILE}" -o "${mhap_file}" >> "${sample_log}" 2>&1
        tabix -b 2 -e 3 -p bed "${mhap_file}" >> "${sample_log}" 2>&1
    fi
    
    mv "${work_dir}"/*_fastqc.html "${output_batch_dir}/" 2>/dev/null
    rm -rf "${work_dir}"
    cleanup_redundant_files "${output_batch_dir}" "${sample_basename}" "${sample_log}"
    
    mark_sample_done "${sample_name}"
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local elapsed_fmt=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
    log_success "样本处理完成: ${sample_name}，耗时: ${elapsed_fmt}" | tee -a "${sample_log}"
    echo "${sample_name}" >> "${SUCCESS_LOG}"
}

process_batch() {
    local batch_dir=$1
    local batch_name=$(basename "$batch_dir")
    
    log_info "========== 处理批次: ${batch_name} =========="
    local output_batch_dir="${OUTPUT_DIR}/${batch_name}"
    mkdir -p "${output_batch_dir}"
    local batch_log_dir="${LOG_DIR}/batch_logs/${batch_name}"
    mkdir -p "${batch_log_dir}"
    
    local r1_files=($(find "${batch_dir}" -maxdepth 1 -name "*_R1_001.fastq.gz" | grep -v "Undetermined" | sort))
    
    if [ ${#r1_files[@]} -eq 0 ]; then
        return 0
    fi
    
    # 【修复核心代码：兼容旧版Bash的并发控制】
    for r1_file in "${r1_files[@]}"; do
        local r2_file="${r1_file/_R1_001.fastq.gz/_R2_001.fastq.gz}"
        local sample_basename=$(basename "$r1_file" "_R1_001.fastq.gz")
        local sample_log="${batch_log_dir}/${sample_basename}.log"
        
        # 检查当前后台运行的 process_sample 任务数量
        # 如果正在运行的数量 >= PARALLEL_SAMPLES，就暂停等待
        while [ $(jobs -pr | wc -l) -ge $PARALLEL_SAMPLES ]; do
            sleep 10
        done
        
        # 提交新任务到后台
        process_sample "${batch_dir}" "${r1_file}" "${r2_file}" "${output_batch_dir}" "${sample_log}" &
    done
    
    # 等待该批次所有后台任务完成
    wait
    
    log_info "========== 批次完成: ${batch_name} =========="
}

main() {
    log_info "==================== WGBS Pipeline 开始 ===================="
    init_directories
    check_tools
    prepare_bismark_index
    
    local wgbs_batches=($(find "${RAW_DATA_DIR}" -maxdepth 1 -type d -name "*BWGBS*" | sort))
    
    for batch_dir in "${wgbs_batches[@]}"; do
        process_batch "${batch_dir}"
        if [ $TEST_MODE -eq 1 ]; then
            local processed_count=$(wc -l < "${PROGRESS_FILE}" 2>/dev/null || echo 0)
            if [ $processed_count -ge $TEST_SAMPLES_LIMIT ]; then
                break
            fi
        fi
    done
    
    log_info "==================== Pipeline 完成 ===================="
}

main "$@"