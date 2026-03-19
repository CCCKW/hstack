import numpy as np
import scipy.sparse as sp
import os

def compute_insulation_score(matrix, window=10, normalize=True):
    if sp.issparse(matrix):
        dense_mat = matrix.toarray()
    else:
        dense_mat = np.asarray(matrix)
        
    n = dense_mat.shape[0]
    scores = np.full(n, np.nan, dtype=np.float32)
    
    if n <= 2 * window:
        return scores
    
    P = np.zeros((n + 1, n + 1), dtype=np.float32)
    P[1:, 1:] = np.cumsum(np.cumsum(dense_mat, axis=0), axis=1)
    
    valid_i = np.arange(window, n - window)
    
    r1 = valid_i - window
    r2 = valid_i
    c1 = valid_i
    c2 = valid_i + window
    
    sum_squares = P[r2, c2] - P[r1, c2] - P[r2, c1] + P[r1, c1]
    scores[valid_i] = sum_squares
    
    if normalize:
        valid_scores = scores[~np.isnan(scores)]
        valid_mean = np.mean(valid_scores)
        print("Valid mean:", valid_mean)
        if valid_mean > 0:
            scores = np.log2(scores / valid_mean + 1e-5)
        else:
            scores = np.zeros_like(scores)

    return scores

p='/Users/ckw/warehouse/metacell/stark/test_output/temp_50000/raw/chr1_sparse_adj.npy'
mats=np.load(p, allow_pickle=True)
mat = mats[0]

scores = compute_insulation_score(mat, window=6)
print("NaN count:", np.isnan(scores).sum())
print("Total size:", len(scores))
print("First few valid scores:", scores[~np.isnan(scores)][:10])
print("Is matrix sparse?", sp.issparse(mat))
print("Matrix nnz:", mat.nnz)

dense_mat = mat.toarray()
print("max value in dense mat:", np.max(dense_mat))

