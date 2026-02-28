import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, cache=True)
def _numba_update_A_incremental(A, G_recon, T1, T2, lambda_sparse, lambda_balance, 
                                max_iters, recompute_interval):
    """
    使用 Frank-Wolfe 算法并行更新分配矩阵 A
    """
    K, N = A.shape
    current_sizes = np.zeros(K)
    for k in range(K):
        s = 0.0
        for n in range(N):
            s += A[k, n]
        current_sizes[k] = s
    avg_size = N / K
    
    for t in range(max_iters):
        gamma = 2.0 / (t + 2.0)
        one_minus_gamma = 1.0 - gamma
        min_indices = np.zeros(N, dtype=np.int32)
        
        for j in prange(N):
            min_val = np.inf
            min_idx = -1
            for k in range(K):
                # 计算梯度：重构误差梯度 + 稀疏正则 + 平衡正则梯度
                bal_grad = lambda_balance * (current_sizes[k] - avg_size)
                val = G_recon[k, j] + lambda_sparse + bal_grad
                if val < min_val:
                    min_val = val
                    min_idx = k
            min_indices[j] = min_idx
            
        E_sizes = np.zeros(K)
        for j in range(N):
            idx = min_indices[j]
            E_sizes[idx] += 1.0
            
        for k in range(K):
            current_sizes[k] = one_minus_gamma * current_sizes[k] + gamma * E_sizes[k]
            
        for j in prange(N):
            idx_E = min_indices[j] 
            for k in range(K):
                val_E = 1.0 if k == idx_E else 0.0
                A[k, j] = one_minus_gamma * A[k, j] + gamma * val_E
                # 增量更新梯度矩阵 G_recon，避免每次完全重算
                t1_term = 2.0 * (T1[k, idx_E] - T2[k, j])
                G_recon[k, j] = one_minus_gamma * G_recon[k, j] + gamma * t1_term
                
        if (t + 1) % recompute_interval == 0:
            for j in prange(N):
                for k in range(K):
                    dot = 0.0
                    for l in range(K):
                        dot += T1[k, l] * A[l, j]
                    G_recon[k, j] = 2.0 * (dot - T2[k, j])
    return A