import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import stark as sk

def main():
    # ==========================================
    # 0. 初始化核心数据对象 HData
    # ==========================================
    print(">>> 初始化 HData 对象...")
    resolutions = [ 50000, 100000,500000, 1000000,]
    hdata = sk.HData(
        data_dir="/Users/ckw/warehouse/metacell/data/test_700_snm3c",
        output_dir="/Users/ckw/warehouse/metacell/stark/test_output",
        genome_reference_path="/Users/ckw/warehouse/metacell/hg19.fa.chrom.sizes",
        chrom_list=[f"chr{i}" for i in range(1, 23)],
        resolutions=resolutions
    )

    # ==========================================
    # 1. 执行数据预处理与加载
    # ==========================================
    print("\n>>> 1. 执行数据预处理与加载...")
    sk.pp.process_and_load(
        hdata, 
        force_process=True, 
        cpu_num=10, 
        gpu_num=8
    )

    lb = []
    path = '/Users/ckw/warehouse/metacell/data/test_700_snm3c'
    for val in os.listdir(path):
        if val.endswith('.pairs'):
            lb.append(val.split('.pairs')[0].split('_')[1])
    hdata.obs['label'] = lb


    # ==========================================
    # 2. 推荐 MetaCell 数量 (带原有参数)
    # ==========================================
    print("\n>>> 2. 推荐 MetaCell 数量...")
    min_depth = 10000000
    max_depth = 100000000
    print(f"推荐 MetaCell 数量的目标测序深度范围: {min_depth:.2f} - {max_depth:.2f}")
    min_k, max_k = sk.tl.recommend_metacell_num(
        hdata,
        target_depth_min=min_depth,  # 下限 20M
        target_depth_max=max_depth ,  # 上限 40M
        resolution_param=2.0,     # Leiden resolution
        n_neighbors=15,
        ref_view=1000000
    )

    result = {}
    pbar = tqdm(range(15,75), desc="MetaCell 数量", unit="num")
    for num in pbar:
        # ==========================================
        # 3. 初始化模型参数 (还原所有原有超参数)
        # ==========================================
        print("\n>>> 3. 初始化模型参数...")
        sk.tl.init_model(
        hdata, 
        n_metacells=num,            # 目标 MetaCell 数量
        lambda_sparse=0.0000,      # 稀疏惩罚
        lambda_balance=0.01,       # 平衡惩罚
        lambda_consistency=0.001,  # 一致性惩罚
        max_iter=100,              # 最大迭代次数
        # --- scHi-C 深度优化参数 ---
        min_size_threshold=0.002,  # 重生阈值
        respawn_interval=10,       # 检查频率
        split_metric='pca'         # 分裂准则
        )


        # ==========================================
        # 4. 计算核矩阵
        # ==========================================
        print("\n>>> 4. 计算核矩阵...")
        sk.tl.compute_kernels(hdata)
        # ==========================================
        # 5. 初始化 Waypoint (还原原有初始化参数)
        # ==========================================
        print("\n>>> 5. 初始化 Waypoints (K-Means++)...")
        sk.tl.initialize_waypoints(
        hdata, 
        seed=32, 
        n_micro_clusters=100,       # 对应原代码中的 30
        ref_view_res=500000
        )



        # ==========================================
        # 6. 核心拟合优化
        # ==========================================
        print("\n>>> 6. 开始模型拟合优化...")
        sk.tl.fit(hdata, n_threads=10)
        # 执行评估
        purity_df, metrics = sk.tl.evaluate(hdata, hdata.obs['label'])



        print("\n🎉 全流程运行完毕！")
        result[num] = metrics
    # save result
    np.save('rbs_sk.npy', result)

if __name__ == "__main__":
    main()
