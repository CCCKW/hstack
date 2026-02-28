import os
import subprocess
import concurrent.futures
from tqdm import tqdm

def _single_pair_to_cool(args):
    """
    内部 worker 函数：将单一 pairs 文件转换为 .cool。
    """
    pair_path, cool_path, chrom_sizes, res = args
    cmd = [
        "cooler", "cload", "pairs",
        "-c1", "2", "-p1", "3", "-c2", "4", "-p2", "5",
        f"{chrom_sizes}:{res}",
        "-", 
        cool_path
    ]
    try:
        with open(pair_path, 'rb') as f_in:
            subprocess.run(cmd, stdin=f_in, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] 转换 Cool 失败 {os.path.basename(pair_path)}: {e.stderr.decode()}")
        return False

def _single_cool_to_mcool(args):
    """
    内部 worker 函数：将 .cool 转换为 .mcool (包含多分辨率)。
    """
    cool_path, mcool_path, resolutions_str = args
    # cooler zoomify 命令，-r 参数接收以逗号分隔的分辨率列表
    cmd = [
        "cooler", "zoomify",
        "-r", resolutions_str,
        "-o", mcool_path,
        cool_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] 转换 MCool 失败 {os.path.basename(cool_path)}: {e.stderr.decode()}")
        return False

def pairs_to_cool(hdata, resolution=10000, n_jobs=10, verbose=True):
    """
    将合并后的 Metacell pairs 文件批量转换为指定分辨率的 .cool 文件。
    """
    pair_dir = os.path.join(hdata.output_dir, "merge", "pair")
    cool_dir = os.path.join(hdata.output_dir, "merge", "cool", str(resolution))
    
    if not os.path.exists(pair_dir):
        raise ValueError(f"未找到合并后的 pairs 目录: {pair_dir}")
    if not os.path.exists(cool_dir):
        os.makedirs(cool_dir, exist_ok=True)
        
    pair_files = [f for f in os.listdir(pair_dir) if f.endswith(".pairs")]
    tasks = []
    for f in pair_files:
        pair_path = os.path.join(pair_dir, f)
        cool_path = os.path.join(cool_dir, f.replace(".pairs", ".cool"))
        if not os.path.exists(cool_path):
            tasks.append((pair_path, cool_path, hdata.genome_reference_path, resolution))

    if tasks and verbose:
        print(f"开始转换为 .cool (分辨率: {resolution}, 并行数: {n_jobs})...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            list(tqdm(executor.map(_single_pair_to_cool, tasks), total=len(tasks), desc=f"Pairs -> Cool {resolution}"))
    
    return cool_dir

def cool_to_mcool(hdata, base_resolution=10000, resolutions=None, n_jobs=10, verbose=True):
    """
    将指定分辨率的 .cool 文件转换为包含多分辨率的 .mcool 文件。
    
    参数:
    - hdata: HData 对象
    - base_resolution: 作为输入的 .cool 文件的分辨率
    - resolutions: 目标多分辨率列表。如果不传，默认为 base_resolution + hdata.resolutions
    """
    cool_dir = os.path.join(hdata.output_dir, "merge", "cool", str(base_resolution))
    mcool_dir = os.path.join(hdata.output_dir, "merge", "mcool")
    
    if not os.path.exists(cool_dir):
        raise ValueError(f"基础 Cool 目录不存在: {cool_dir}")
    if not os.path.exists(mcool_dir):
        os.makedirs(mcool_dir, exist_ok=True)

    # 确定要生成的分辨率列表
    if resolutions is None:
        # 默认 = 10000 + 模型所用的分辨率
        res_set = {base_resolution} | set(hdata.resolutions)
        # 排序并转为字符串
        resolutions = sorted(list(res_set))
    
    res_str = ",".join(map(str, resolutions))
    
    cool_files = [f for f in os.listdir(cool_dir) if f.endswith(".cool")]
    tasks = []
    for f in cool_files:
        cool_path = os.path.join(cool_dir, f)
        mcool_path = os.path.join(mcool_dir, f.replace(".cool", ".mcool"))
        if not os.path.exists(mcool_path):
            tasks.append((cool_path, mcool_path, res_str))

    if not tasks:
        if verbose: print(f"所有 .mcool 文件已存在，跳过。")
        return

    if verbose:
        print(f"开始转换为 .mcool (包含分辨率: {res_str}, 并行数: {n_jobs})...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        list(tqdm(executor.map(_single_cool_to_mcool, tasks), total=len(tasks), desc="Cool -> MCool"))

    if verbose:
        print(f"✅ 转换完成！MCool 文件已储存至: {mcool_dir}")