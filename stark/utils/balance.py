import os
import cooler
import concurrent.futures
from tqdm.auto import tqdm

def is_cooler_balanced(clr, store_name='weight'):
    """
    检查 cooler 对象是否已经完成了平衡。
    优先调用 cooltools 的原生方法，若未安装或报错，则降级为直接检查 bins 里的 columns。
    """
    try:
        import cooltools.lib
        # 兼容用户提供的 cooltools 检查逻辑
        return cooltools.lib.is_cooler_balanced(clr)
    except ImportError:
        # Fallback: 最稳妥的原生 Cooler 检查方法
        return store_name in clr.bins().columns

def _balance_single_mcool(args):
    """
    内部 Worker 函数：负责对单一 Metacell 的 mcool 文件下的指定分辨率进行 Balance。
    """
    m_id, mcool_path, resolution, kwargs = args
    
    # 构建指定分辨率的 cooler URI
    uri = f"{mcool_path}::/resolutions/{resolution}"
    
    try:
        clr = cooler.Cooler(uri)
        store_name = kwargs.get('store_name', 'weight')
        
        # 检查是否已平衡，避免重复计算
        if not is_cooler_balanced(clr, store_name):
            # 存盘标志始终为 True
            kwargs['store'] = True 
            cooler.balance_cooler(clr, **kwargs)
            return m_id, True, "Success"
        else:
            return m_id, True, "Already balanced"
            
    except Exception as e:
        return m_id, False, str(e)


def balance_metacells(hdata, resolution, n_jobs=10, verbose=True, **balance_kwargs):
    """
    并行对 HData 中所有 Metacell 的 mcool 文件进行矩阵平衡 (Iterative Correction)。
    
    参数:
    - hdata: HData 对象，内部需通过 aggr 模块注册好了 metacell_data['mcool']。
    - resolution: int, 要进行平衡的分辨率 (该分辨率必须存在于 mcool 文件中)。
    - n_jobs: int, 并行工作的进程数，默认为 10。
    - verbose: bool, 是否显示进度条。
    
    cooler.balance_cooler 支持的可选参数 (均可由 kwargs 直接透传传入):
    - cis_only (bool): 仅对染色体内数据进行平衡。
    - trans_only (bool): 仅对染色体间数据进行平衡。
    - ignore_diags (int): 忽略对角线及附近的元素数，默认为 2。
    - mad_max (int): 预处理过滤阈值，丢弃偏差过大的 bins，默认为 5。
    - min_nnz (int): 预处理过滤阈值，丢弃非零元素过少的 bins，默认为 10。
    - min_count (int): 预处理过滤阈值，丢弃边缘和过低的 bins，默认为 0。
    - blacklist (list/array): 显式忽略的 bad bins 的 ID 列表。
    - rescale_marginals (bool): 缩放权重使得行/列和为 1.0，默认为 True。
    - tol (float): 收敛标准 (边缘和向量的方差)，默认为 1e-05。
    - max_iters (int): 最大迭代次数。
    - chunksize (int): 内存分块大小，调大可加速或应对大内存机器。
    - store_name (str): 保存权重的列名，默认为 'weight'。
    """
    if 'mcool' not in hdata.metacell_data or not hdata.metacell_data['mcool']:
        raise ValueError("hdata 中未找到任何 mcool 文件路径。请先运行聚合工具 aggregate_metacell_pairs。")
        
    mcool_dict = hdata.metacell_data['mcool']
    tasks = []
    
    # 构建默认参数 (按照你的需求，给 chunksize 和 max_iters 赋予了激进的默认值)
    default_kwargs = {
        'max_iters': 100,
        'chunksize': 1000000000000000, 
        'store_name': 'weight'
    }
    
    # 覆盖用户自定义传入的其他 balance 参数
    default_kwargs.update(balance_kwargs)
    
    for m_id, mcool_path in mcool_dict.items():
        if os.path.exists(mcool_path):
            tasks.append((m_id, mcool_path, resolution, default_kwargs.copy()))
        else:
            if verbose: 
                print(f"[Warning] 找不到 Metacell {m_id} 的 mcool 文件: {mcool_path}")

    if not tasks:
        print("未找到需要处理的 mcool 文件。")
        return
        
    if verbose:
        print(f"\n>>> 开始对 {len(tasks)} 个 Metacell 执行矩阵平衡 (Resolution: {resolution}, 并发数: {n_jobs})")
        
    success_count = 0
    # 由于各个 cooler 相互独立，最外层使用 ProcessPool 并发能彻底打满多核 CPU
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(_balance_single_mcool, tasks), 
                            total=len(tasks), 
                            desc=f"Balancing {resolution}", 
                            disable=not verbose))
        
        for m_id, success, msg in results:
            if success:
                success_count += 1
            else:
                if verbose: 
                    print(f"\n[Error] Metacell {m_id} 平衡失败: {msg}")

    if verbose:
        print(f"✅ 矩阵平衡完成！(成功/总数: {success_count}/{len(tasks)})")