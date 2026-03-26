import os
import numpy as np
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["NUMBA_THREADING_LAYER"
] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
import stark as sk


# ==========================================
# 0. 初始化核心数据对象 HData
# ==========================================
print(">>> 初始化 HData 对象...")
resolutions = [ 50000,500000, 1000000,]
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
lb = ['ExcNeuron' if x in ['L23', 'L4', 'L5', 'L6'] else x for x in lb]
hdata.obs['label'] = lb

result = pd.DataFrame(columns=['mean_purity', 'acc', 'global_score', 'wcos', 'hwis'])

for i in range(11, 40):
    print('########', i, '########')
    # # ==========================================
    # # 3. 初始化模型参数 (还原所有原有超参数)
    # # ==========================================
    # print("\n>>> 3. 初始化模型参数...")
    sk.tl.init_model(
        hdata, 
        n_metacells=i, # 25,            # 目标 MetaCell 数量
        lambda_balance=0.5,       # 平衡惩罚
        lambda_consistency=0.010,  # 一致性惩罚
    adaptive_weight=True,
        max_iter=100,              # 最大迭代次数
        # --- scHi-C 深度优化参数 ---
        min_size_threshold=0.002,  # 重生阈值
        respawn_interval=100,       # 检查频率
        split_metric='pca',         # 分裂准则
        weight_method='consensus',
        lambda_ortho=0.001
        # weight_momentum=0.99,
    )
        # ==========================================
    # 4. 计算核矩阵
    # ==========================================
    print("\n>>> 4. 计算核矩阵...")
    sk.tl.compute_kernels(hdata, )
    # sk.tl.compute_kernels(hdata, use_ps=True)
    # sk.tl.compute_kernels(hdata, use_ps=False)÷
    # ==========================================
    # 5. 初始化 Waypoint (还原原有初始化参数)
    # ==========================================
    print("\n>>> 5. 初始化 Waypoints (K-Means++)...")
    sk.tl.initialize_waypoints(
        hdata, 
        data_type='pca', 
        seed=32, 
        n_micro_clusters=300,       # 对应原代码中的 30
        ref_view_res=1000000
    )
    
    
    # ==========================================
    # 6. 核心拟合优化
    # ==========================================
    print("\n>>> 6. 开始模型拟合优化...")
    sk.tl.fit(hdata, n_threads=10)

    # 执行评估
    purity_df, metrics = sk.tl.evaluate(hdata, hdata.obs['label'])


    print("\n🎉 全流程运行完毕！")
    vals = np.array(metrics.values())
    print(vals)
    result.loc[i] = vals

result.to_csv('stark.csv')