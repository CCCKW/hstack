import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
from tqdm.auto import tqdm
import scanpy as sc
from multiprocessing import cpu_count

NUM_CORES = cpu_count()

def kth_neighbor_distance(distances, k, i):
    """返回到第 k 个最近邻的距离"""
    row_as_array = distances[i, :].toarray().ravel()
    num_nonzero = np.sum(row_as_array > 0)
    kth_neighbor_idx = np.argsort(np.argsort(-row_as_array)) == num_nonzero - k
    return np.linalg.norm(row_as_array[kth_neighbor_idx])

def rbf_for_row(G, data, median_distances, i):
    """为数据矩阵的每一行计算径向基函数 (RBF) 核"""
    row_as_array = G[i, :].toarray().ravel()
    numerator = np.sum(np.square(data[i, :] - data), axis=1, keepdims=False)
    denominator = median_distances[i] * median_distances
    full_row = np.exp(-numerator / denominator)
    masked_row = np.multiply(full_row, row_as_array)
    return lil_matrix(masked_row)

class SEACellGraph:
    """内部化 SEACell 图构建逻辑，替代原第三方库"""

    def __init__(self, ad, build_on="X_pca", n_cores: int = -1, verbose: bool = False):
        self.n, self.d = ad.obsm[build_on].shape
        self.indices = np.array(range(self.n))
        self.ad = ad
        self.build_on = build_on
        self.knn_graph = None
        self.sym_graph = None
        self.num_cores = n_cores if n_cores != -1 else NUM_CORES
        self.M = None  
        self.verbose = verbose

    def rbf(self, k: int = 15, graph_construction="union"):
        """初始化自适应带宽 RBF 核"""
        if self.verbose:
            print("正在使用 Scanpy 计算 kNN 图 ...")

        sc.pp.neighbors(self.ad, use_rep=self.build_on, n_neighbors=k, knn=True)
        knn_graph_distances = self.ad.obsp["distances"]

        knn_graph = knn_graph_distances.copy()
        knn_graph[knn_graph != 0] = 1
        knn_graph.setdiag(1)

        self.knn_graph = knn_graph
        if self.verbose:
            print("正在计算自适应带宽核的半径...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median = k // 2
            median_distances = parallel(
                delayed(kth_neighbor_distance)(knn_graph_distances, median, i)
                for i in tqdm(range(self.n), disable=not self.verbose)
            )

        median_distances = np.array(median_distances)

        if self.verbose:
            print(f"使图对称化 (策略: {graph_construction})...")

        if graph_construction == "union":
            sym_graph = (knn_graph + knn_graph.T > 0).astype(float)
        elif graph_construction in ["intersect", "intersection"]:
            knn_graph = (knn_graph > 0).astype(float)
            sym_graph = knn_graph.multiply(knn_graph.T)
        else:
            raise ValueError("无效的 graph_construction 参数。")

        self.sym_graph = sym_graph
        if self.verbose:
            print("正在计算 RBF 核...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(
                delayed(rbf_for_row)(
                    sym_graph, self.ad.obsm[self.build_on], median_distances, i
                )
                for i in tqdm(range(self.n), disable=not self.verbose)
            )

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in range(self.n):
            similarity_matrix[i] = similarity_matrix_rows[i]

        self.M = similarity_matrix.tocsr()
        return self.M