import os
import json
import gzip
import shutil
import pandas as pd
import concurrent.futures
from tqdm import tqdm

def _merge_single_metacell(args):
    """
    内部 worker 函数：负责单个 Metacell 的文件合并逻辑。
    """
    m_id, indices, all_files, save_dir = args
    target_path = os.path.join(save_dir, f"metacell_{m_id}.pairs")
    
    with open(target_path, 'wb') as f_out:
        header_written = False
        for idx in indices:
            src_path = all_files[idx]
            is_gz = src_path.endswith('.gz')
            opener = gzip.open(src_path, 'rb') if is_gz else open(src_path, 'rb')
            with opener as f_in:
                if not header_written:
                    shutil.copyfileobj(f_in, f_out)
                    header_written = True
                else:
                    for line in f_in:
                        if not line.startswith(b'#'):
                            f_out.write(line)
    return m_id, target_path

def aggregate_metacell_pairs(hdata, n_jobs=10, force_aggregate=False, 
                             convert_to_cool=False, convert_to_mcool=False,
                             resolution=10000, mcool_resolutions=None, verbose=True):
    """
    并行合并 pairs 并在 hdata 中注册 Metacell 级别的信息。
    采用日志状态记录确保重跑时的分配与参数校验完全一致。
    """
    if 'metacell' not in hdata.obs:
        raise ValueError("hdata.obs 中未发现 'metacell' 标签。请确保已运行拟合步骤。")

    # --- 1. 更新 hdata.metacells 基础信息 (作为兜底机制) ---
    if 'total_depth' not in hdata.metacells.columns or force_aggregate:
        if verbose: print(">>> 未检测到基础属性或被强制刷新，正在统计 Metacell 属性...")
        meta_stats = hdata.obs.groupby('metacell').agg({
            'depth': ['sum', 'count', 'mean']
        })
        meta_stats.columns = ['total_depth', 'cell_count', 'mean_depth']
        
        if 'label' in hdata.obs.columns:
            def get_dominant(x): return x.value_counts().index[0]
            meta_stats['dominant_label'] = hdata.obs.groupby('metacell')['label'].apply(get_dominant)
            
        if hdata.metacells.empty:
            hdata.metacells = meta_stats
        else:
            for col in meta_stats.columns:
                hdata.metacells[col] = meta_stats[col]

    log_dir = os.path.join(hdata.output_dir, "merge")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "aggr_log.json")
    save_dir = os.path.join(log_dir, "pair")

    # --- 2. 状态校验 (读取/对比 Log) ---
    current_allocation = {str(k): int(v) for k, v in hdata.obs['metacell'].items()}
    current_params = {
        "resolution": int(resolution),
        "mcool_resolutions": sorted([int(r) for r in mcool_resolutions]) if mcool_resolutions else [],
        "convert_to_cool": bool(convert_to_cool),
        "convert_to_mcool": bool(convert_to_mcool)
    }
    
    # 【核心修复】强制同步 hdata.resolutions
    # 防止底层的 pair2cool 等模块直接读取 hdata 原生属性而忽略这里的参数
    if mcool_resolutions:
        hdata.resolutions = current_params["mcool_resolutions"]

    allocation_changed = True
    params_changed = True
    saved_state = {}

    if not force_aggregate and os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                saved_state = json.load(f)
            
            saved_pairs = saved_state.get('output_paths', {}).get('pairs', {})
            
            # 校验 1：细胞归属是否变化 & pairs 文件是否完整存在
            if saved_state.get("allocation") == current_allocation and saved_pairs and all(os.path.exists(p) for p in saved_pairs.values()):
                allocation_changed = False
                # 既然分配没变，把硬盘上已有的 pairs 路径恢复到内存，供下游使用
                hdata.metacell_data['pairs'] = {int(k): v for k, v in saved_pairs.items()}
                
                # 顺便继承历史存在的 cool/mcool 路径（即使稍后参数改变，老分辨率的数据也不该丢失）
                if 'cool' in saved_state.get('output_paths', {}):
                    for res, d in saved_state['output_paths']['cool'].items():
                        if res not in hdata.metacell_data['cool']:
                            hdata.metacell_data['cool'][str(res)] = {int(k): v for k, v in d.items()}
                if 'mcool' in saved_state.get('output_paths', {}):
                    # 避免直接覆盖内存中的现有数据
                    for k, v in saved_state['output_paths']['mcool'].items():
                        hdata.metacell_data['mcool'].setdefault(int(k), v)

            # 校验 2：分辨率等参数是否变化
            if saved_state.get("parameters") == current_params:
                params_changed = False

            if not allocation_changed and not params_changed:
                if verbose: print(">>> 检测到分配映射与聚合参数均未改变，且文件完整，跳过所有重复生成。")

        except Exception as e:
            if verbose: print(f">>> 读取或解析校验日志失败，将重新合并生成: {e}")

    # --- 3. 执行 Pairs 合并 ---
    if allocation_changed:
        if verbose and os.path.exists(log_path) and not force_aggregate:
            print(">>> 分配发生改变或被强制重跑，正在重新聚合并覆盖历史 pairs 文件...")

        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        pair_paths = {}
        all_files = []
        for val in sorted(os.listdir(hdata.data_dir)):
            if val.endswith('.pairs') or val.endswith('.pairs.gz'):
                all_files.append(os.path.join(hdata.data_dir, val))
        
        metacell_groups = hdata.obs.groupby('metacell').groups
        tasks = [(m_id, indices, all_files, save_dir) for m_id, indices in metacell_groups.items()]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(_merge_single_metacell, tasks), total=len(tasks), desc="Aggregating Pairs"))
            for m_id, p in results: pair_paths[m_id] = p
    
        hdata.metacell_data['pairs'] = pair_paths
    else:
        if verbose and params_changed and (convert_to_cool or convert_to_mcool):
            print(">>> 细胞分配未变，跳过 Pairs 合并。检测到分辨率等参数改变，直接处理 Cool/MCool 转换...")

    # --- 4. 自动转换并注册 Cool/MCool ---
    if (allocation_changed or params_changed) and (convert_to_cool or convert_to_mcool):
        from .pair2cool import pairs_to_cool, cool_to_mcool
        
        old_params = saved_state.get("parameters", {})
        
        if convert_to_cool:
            cool_dir = pairs_to_cool(hdata, resolution=resolution, n_jobs=n_jobs, verbose=verbose)
            if str(resolution) not in hdata.metacell_data['cool']:
                hdata.metacell_data['cool'][str(resolution)] = {}
            for f in os.listdir(cool_dir):
                if f.endswith('.cool'):
                    m_id = int(f.split('_')[1].split('.')[0])
                    hdata.metacell_data['cool'][str(resolution)][m_id] = os.path.join(cool_dir, f)

        if convert_to_mcool:
            mcool_dir = os.path.join(hdata.output_dir, "merge", "mcool")
            
            # --- 核心修复：清理旧的 mcool 文件强制重新生成 ---
            old_mcool_res = old_params.get("mcool_resolutions", [])
            curr_mcool_res = current_params["mcool_resolutions"]
            res_changed = old_params.get("resolution") != current_params["resolution"]
            mcool_res_changed = old_mcool_res != curr_mcool_res
            
            if allocation_changed or mcool_res_changed or res_changed or force_aggregate:
                if verbose and not allocation_changed and mcool_res_changed:
                    print(f">>> 检测到 mcool 分辨率列表发生改变 (新列表: {curr_mcool_res})，正在彻底清理缓存的旧 mcool 文件以强制重新生成...")
                
                # 1. 彻底清空目标目录，拒绝任何忽略报错而导致的残留
                if os.path.exists(mcool_dir):
                    shutil.rmtree(mcool_dir)
                os.makedirs(mcool_dir, exist_ok=True)
                
                # 2. 【关键】彻底清理可能残留在 cool_dir 中的 .mcool 文件
                # cooler zoomify 默认在输入文件同级目录生成，若有历史残留会导致它直接跳过生成
                cool_dir = os.path.join(hdata.output_dir, "merge", "cool")
                if os.path.exists(cool_dir):
                    for f in os.listdir(cool_dir):
                        if f.endswith('.mcool'):
                            try:
                                os.remove(os.path.join(cool_dir, f))
                            except OSError:
                                pass
                                
                # 3. 清空内存中失效的路径引用
                hdata.metacell_data['mcool'] = {}
            # ------------------------------------------------

            cool_to_mcool(hdata, base_resolution=resolution, resolutions=mcool_resolutions, n_jobs=n_jobs, verbose=verbose)
            
            for f in os.listdir(mcool_dir):
                if f.endswith('.mcool'):
                    m_id = int(f.split('_')[1].split('.')[0])
                    hdata.metacell_data['mcool'][m_id] = os.path.join(mcool_dir, f)

    # --- 5. 写入最新的状态日志 ---
    if allocation_changed or params_changed:
        current_state = {
            "allocation": current_allocation,
            "parameters": current_params,
            "output_paths": {
                "pairs": hdata.metacell_data.get('pairs', {}),
                "cool": hdata.metacell_data.get('cool', {}),
                "mcool": hdata.metacell_data.get('mcool', {})
            }
        }
        with open(log_path, 'w') as f:
            json.dump(current_state, f, indent=4)

    if verbose: print(f"\n✅ 聚合流程完成。Metacell 文件路径与信息已存入 hdata。\n{hdata}")

def aggregate_metacell_mat(hdata, force_aggregate=False, verbose=True):
    """
    直接从 hdata.views_mat 中批量合并所有已初始化分辨率的单细胞稀疏矩阵，生成 Metacell 级别的稀疏矩阵。
    独立于 pairs 聚合流程，生成的结果将储存在 hdata.metacell_data['mat'][str(resolution)] 中。
    运行完毕后将打印可供 base_on='mat' 可视化的分辨率列表。
    """
    if 'metacell' not in hdata.obs:
        raise ValueError("hdata.obs 中未发现 'metacell' 标签。请确保已运行拟合步骤。")

    if not hdata.views_mat:
        raise ValueError("hdata.views_mat 为空，请先运行 sk.pp.process_and_load。")

    if 'mat' not in hdata.metacell_data:
        hdata.metacell_data['mat'] = {}

    metacell_groups = hdata.obs.groupby('metacell').groups
    chrom_list = hdata.chrom_list

    # 预计算位置索引（与分辨率无关，只算一次）
    pos_idx_map = {
        m_id: [hdata.obs.index.get_loc(idx) for idx in indices_list]
        for m_id, indices_list in metacell_groups.items()
    }

    completed_resolutions = []

    for resolution, mat_dict in hdata.views_mat.items():
        str_res = str(resolution)

        if str_res in hdata.metacell_data['mat'] and not force_aggregate:
            if verbose:
                print(f"⏭️  分辨率 {resolution} 已存在聚合结果，跳过 (使用 force_aggregate=True 强制重跑)。")
            completed_resolutions.append(resolution)
            continue

        # 校验染色体完整性
        missing = [c for c in chrom_list if c not in mat_dict]
        if missing:
            print(f"⚠️  分辨率 {resolution} 缺少染色体 {missing} 的矩阵数据，跳过该分辨率。")
            continue

        hdata.metacell_data['mat'][str_res] = {}
        all_res_dict = hdata.metacell_data['mat'][str_res]

        for m_id, pos_indices in tqdm(pos_idx_map.items(), desc=f"Aggregating @ {resolution}"):
            m_id_dict = {}
            for chrom in chrom_list:
                chrom_mats = mat_dict[chrom]
                sum_mat = chrom_mats[pos_indices[0]].copy()
                for idx in pos_indices[1:]:
                    sum_mat += chrom_mats[idx]
                m_id_dict[chrom] = sum_mat
            all_res_dict[m_id] = m_id_dict

        completed_resolutions.append(resolution)

    if verbose:
        print(f"\n✅ 全量聚合完成！")
        print(f"   可通过 base_on='mat' 方式可视化的分辨率: {sorted(completed_resolutions)}")
        if completed_resolutions:
            print(f"   调用示例: sk.pl.plot_metacell_heatmap(hdata, ..., resolution={sorted(completed_resolutions)[0]}, base_on='mat')")
