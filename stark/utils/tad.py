import numpy as np




def compute_insulation_score(sparse_matrix, window=10):
    """
    对单个细胞的稀疏接触矩阵计算绝缘分数向量
    window: 滑动窗口大小（bin数）
    """
    n = sparse_matrix.shape[0]
    scores = np.zeros(n)
    dense = sparse_matrix.toarray()
    for i in range(window, n - window):
        scores[i] = dense[i-window:i, i:i+window].mean()
    # 归一化：相对于全局均值
    mean = scores[scores > 0].mean()
    scores = scores / (mean + 1e-8)
    return np.log1p(scores)