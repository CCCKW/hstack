import numpy as np
import scipy.sparse as sp

def compute_insulation_score(matrix, window=10, normalize=True):
    """
    对单个细胞的单条染色体接触矩阵计算绝缘分数向量 (Insulation Score, IS).
    采用了快速的前缀和 (Integral Image) 算法以提升性能，能够即时计算上万个细胞。
    
    参数:
    matrix (scipy.sparse.spmatrix or numpy.ndarray): N x N 维度的单条染色体接触矩阵。
    window (int): 滑动窗口大小 (bin 数)，推荐物理距离 300kb~500kb。
                  例如 50kb 分辨率下，推荐设定 window=6 ~ 10。
    normalize (bool): 是否对当前细胞的绝缘分数执行细胞内标准化。
    
    返回:
    scores (np.ndarray): 一维绝缘分数数组，长度为 N。两端边界位置处得分为 np.nan。
    """
    if sp.issparse(matrix):
        dense_mat = matrix.toarray()
    else:
        dense_mat = np.asarray(matrix)
        
    n = dense_mat.shape[0]
    scores = np.full(n, np.nan, dtype=np.float32)
    
    if n <= 2 * window:
        # 如果染色体长度过短，不足以构成一个完整的滑动窗口
        return scores
    
    # 构建二维前缀和积分图 (Integral Image)，将时间复杂度降为 O(1) 滑动求和
    # P[i, j] 记录了 matrix[0..i-1, 0..j-1] 内所有元素的累加和
    P = np.zeros((n + 1, n + 1), dtype=np.float32)
    # 利用 numpy.cumsum 加速计算
    P[1:, 1:] = np.cumsum(np.cumsum(dense_mat, axis=0), axis=1)
    
    # 构造有效滑动索引：i 表示中心 bin，滑动范围从 window 开始，到 n - window 结束
    valid_i = np.arange(window, n - window)
    
    # 绝缘分数的定义：计算中心 bin 的上游 w 个 bin 和下游 w 个 bin 之间的相互作用强度总和
    # 在接触图中对应一个从 (i-window, i) 到 (i, i+window) 的方块。
    r1 = valid_i - window
    r2 = valid_i
    c1 = valid_i
    c2 = valid_i + window
    
    # O(1) 利用积分图获取每个正方形内接触数的总和
    sum_squares = P[r2, c2] - P[r1, c2] - P[r2, c1] + P[r1, c1]
    
    # 将绝缘分数存放到对应的位置
    scores[valid_i] = sum_squares
    
    # --- 细胞内标准化 (Intra-cell Normalization) ---
    if normalize:
        valid_scores = scores[~np.isnan(scores)]
        valid_mean = np.mean(valid_scores)
        
        if valid_mean > 0:
            # 加入 1e-5 以避免极低覆盖度背景中的零除问题，使用 log2 进行对数折叠
            scores = np.log2(scores / valid_mean + 1e-5)
        else:
            # 极少见的边角异常，当前染色体几乎无有效接触
            scores = np.zeros_like(scores)

    return scores


def call_tad_boundaries(insulation_scores, window=5, min_prominence=0.1):
    """
    从全类群假体 (pseudo-bulk) 的标准化一维绝缘分数字典（数组）中探测共识 TAD 边界（谷值点）。
    这非常适合应用在你层次 Metacell 聚合中作为特征筛选（Feature Selection）的前置步骤。
    
    参数:
    insulation_scores (np.ndarray): 一维绝缘得分向量 (包含 np.nan)。
    window (int): 寻找局部最小谷底所需的探测距离。这可以小一点，比如 3-5。
    min_prominence (float): 边界谷底相对于周边高峰和背景的显著深度阈值。
    
    返回:
    boundaries (np.ndarray): 识别出的 TAD 边界所处的 bin 的原始索引数组，用于切选单细胞特征。
    """
    from scipy.signal import find_peaks
    
    # 绝缘分数的谷值（TAD boundaries）即为负向量的峰值 (peaks)
    valid_mask = ~np.isnan(insulation_scores)
    
    # 为了使用 find_peaks，对合法的值进行符号翻转
    inv_scores = np.full_like(insulation_scores, np.nan)
    inv_scores[valid_mask] = -insulation_scores[valid_mask]
    
    # 检测局部峰值（即原始分布极其尖锐的绝缘深谷底）
    valid_inv_scores = inv_scores[valid_mask]
    peaks, properties = find_peaks(
        valid_inv_scores, 
        distance=window, 
        prominence=min_prominence
    )
    
    # 映射回完整的染色体基因组坐标系下的 bin 索引 (1维)
    original_indices = np.where(valid_mask)[0]
    boundaries = original_indices[peaks]
    
    return boundaries