import os
import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial.distance import cdist
from scipy.sparse import issparse, vstack, save_npz, load_npz
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from numba import set_num_threads

from .numba_ops import _numba_update_A_incremental
from .graph import SEACellGraph

from .evaluation import EvaluationMixin  




class MultiViewSEACells():  
    """
    多视图 SEACells 联合优化模型 (内置图构建，移除对第三方 SEACells 的显式依赖)
    融合了完整的评估体系 (EvaluationMixin) 与可视化功能 (PlottingMixin)
    """
    def __init__(self, n_metacells=100, lambda_sparse=0.05, lambda_balance=0.01, 
                 lambda_consistency=0.1, max_iter=100, adaptive_weight=True, 
                 weight_momentum=0.9, max_franke_wolfe_iters=50,
                 n_neighbors=15, # 新增 KNN 参数，替代原有参数透传
                 min_size_threshold=0.002, 
                 respawn_interval=10,
                 split_metric='pca'):
        
        self.n_metacells = n_metacells
        self.lambda_sparse = lambda_sparse
        self.lambda_balance = lambda_balance
        self.lambda_consistency = lambda_consistency
        self.max_iter = max_iter
        self.adaptive_weight = adaptive_weight
        self.weight_momentum = weight_momentum
        self.max_franke_wolfe_iters = max_franke_wolfe_iters
        self.n_neighbors = n_neighbors
        
        # 动态策略配置
        self.min_size_threshold = min_size_threshold
        self.respawn_interval = respawn_interval
        self.split_metric = split_metric
        
        # 内部状态
        self.kernels_computed = False
        self.initialized = False
        self.waypoints = None
        self.kernels = []
        self.view_weights = None
        self.n_cells = 0
        self.views_data = None 
        self.kernel_norms_sq = []
        
        # 评估相关的属性预占位 (支持分步调用，在 calculate_metrics() 中赋值)
        self.global_score_ = None
        self.accuracy_ = None
        self.mean_purity_ = None
        self.purity_df_ = None     # 缓存的纯度指标表
        self._eval_df_cache = None # 内部细胞信息缓存

    def compute_kernels(self, views_data, save_dir=None):
        """步骤1: 计算核矩阵 (带缓存功能)"""
        self.views_data = views_data
        self.n_cells = views_data[0].shape[0]
        self.kernels = [] 
        self.kernel_norms_sq = []
        
        print("=" * 60)
        print(f"步骤1: 构建核矩阵 (Cache Dir: {save_dir if save_dir else 'None'})")
        print("=" * 60)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, view in enumerate(views_data):
            loaded_from_cache = False
            cache_path = None
            if save_dir:
                cache_path = os.path.join(save_dir, f'kernel_view_{i}.npz')
                if os.path.exists(cache_path):
                    print(f"  [View {i}] 发现缓存，正在读取: {cache_path}")
                    try:
                        M = load_npz(cache_path)
                        if M.shape[0] != self.n_cells or M.shape[1] != self.n_cells:
                            print(f"  [警告] 缓存维度 {M.shape} 不匹配，将重新计算。")
                        else:
                            loaded_from_cache = True
                            print(f"  [View {i}] 缓存加载成功。")
                    except Exception as e:
                        print(f"  [警告] 缓存读取失败: {e}，将重新计算。")
            
            if not loaded_from_cache:
                print(f"  [View {i}] 开始内部计算 RBF 核矩阵 (Input: {view.shape})...")
                tmp = ad.AnnData(view)
                tmp.obsm['X_pca'] = view
                
                # 直接使用内置的 SEACellGraph，替代原版第三方包调用
                kernel_model = SEACellGraph(tmp, build_on='X_pca', verbose=False)
                M = kernel_model.rbf(k=self.n_neighbors, graph_construction='union')
                
                if save_dir and cache_path:
                    print(f"  [View {i}] 正在写入缓存...")
                    save_npz(cache_path, M)
            
            self.kernels.append(M)
            norm_sq = sparse_norm(M)**2
            self.kernel_norms_sq.append(norm_sq)
            
            sparsity = (M.toarray() > 0).sum() / M.size if M.shape[0] < 5000 else 'Large Sparse'
            print(f"  -> 核矩阵就绪: Shape={M.shape}, Sparsity={sparsity}")
        
        self.kernels_computed = True
        return self

    def initialize(self, n_threads=None, seed=None, data_type='views', n_micro_clusters=None):
        """步骤2: 初始化 (基于微簇)"""
        if not self.kernels_computed:
            raise RuntimeError("请先运行 compute_kernels。")
            
        if seed is not None:
            np.random.seed(seed)
            
        print("\n" + "=" * 60)
        print(f"步骤2: 参数初始化 (Data: {data_type}, Method: Micro-Clustering)")
        print("=" * 60)
        
        n_views = len(self.kernels)
        self.view_weights = np.ones(n_views) / n_views
        
        if data_type == 'views':
            cluster_data = np.hstack(self.views_data)
        elif data_type == 'kernel':
            print("  正在合并核矩阵用于初始化...")
            cluster_data = vstack(self.kernels).T 
        else:
            raise ValueError("data_type 必须是 'views' 或 'kernel'")

        if n_micro_clusters is None:
            n_micro_clusters = int(self.n_metacells * 2.0)
            
        print(f"  执行 MiniBatchKMeans (k={n_micro_clusters})...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_micro_clusters, 
            random_state=seed, 
            batch_size=1024,
            n_init=3
        ).fit(cluster_data)
        
        micro_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        df_micro = pd.DataFrame({'label': micro_labels})
        size_stats = df_micro['label'].value_counts()
        min_micro_size = max(5, int(self.n_cells * 0.0005)) 
        valid_labels = size_stats[size_stats >= min_micro_size].index.tolist()
        
        if len(valid_labels) < self.n_metacells:
            print("  [警告] 有效微簇少于目标数，强制使用最大的簇。")
            valid_labels = size_stats.index.tolist()[:self.n_metacells]

        final_waypoints = []
        top_k_labels = size_stats[size_stats.index.isin(valid_labels)].index[:self.n_metacells]
        
        for lbl in top_k_labels:
            indices = np.where(micro_labels == lbl)[0]
            if data_type == 'views':
                dists = np.linalg.norm(cluster_data[indices] - centers[lbl], axis=1)
                final_waypoints.append(indices[np.argmin(dists)])
            else:
                final_waypoints.append(np.random.choice(indices))
                
        self.waypoints = np.array(final_waypoints)
        print(f"  最终选中 {len(self.waypoints)} 个 Waypoints")
        
        self.B = np.zeros((self.n_cells, self.n_metacells), dtype=np.float32)
        for i, wp in enumerate(self.waypoints):
            self.B[wp, i] = 1.0
            
        self.A = np.random.random((self.n_metacells, self.n_cells)).astype(np.float32)
        self.A = normalize(self.A, axis=0, norm='l1')
        
        self.initialized = True
        return self

    def fit(self, CellTypes=None, n_threads=None):
        """步骤4: 训练 (带改进的重生逻辑)"""
        self.CellTypes = CellTypes
        if not self.initialized: raise RuntimeError("未初始化")
        if n_threads: set_num_threads(n_threads)

        print("\n" + "=" * 60)
        print(f"步骤4: 联合优化 (Split Metric: {self.split_metric})")
        print("=" * 60)
        
        losses = []
        for iteration in range(self.max_iter):
            self.A = self._updateA_incremental(self.B, self.A, self.view_weights, iteration)
            self.B = self._updateB_incremental(self.A, self.B, self.view_weights)
            
            if iteration > 0 and iteration < (self.max_iter * 0.8) and iteration % self.respawn_interval == 0:
                self._check_and_respawn(iteration)

            if iteration % 10 == 0:
                loss, recon_errors, cons_error, sparse_reg = self._compute_loss_efficient(
                    self.kernels, self.B, self.A, self.view_weights
                )
                losses.append(loss)
                sizes = self.A.sum(axis=1)
                print(f"Iter {iteration:3d} | Loss: {loss:.4f} | Size Range: {sizes.min():.1f}-{sizes.max():.1f}")

            if self.adaptive_weight and iteration > 0:
                self.view_weights = self._update_weights_consensus(
                    self.kernels, self.B, self.view_weights, self.weight_momentum
                )
        
        self.labels = np.array(self.A.argmax(0)).reshape(-1)
        print("\n优化完成")
        return self

    def _check_and_respawn(self, iteration):
        sizes = self.A.sum(axis=1)
        size_threshold = max(1.0, self.n_cells * self.min_size_threshold)
        labels = self.A.argmax(axis=0)
        ref_data = self.views_data[0]
        
        starving_indices = [k for k in range(self.n_metacells) if sizes[k] < size_threshold]
        
        centroids = np.zeros((self.n_metacells, ref_data.shape[1]))
        density_scores = np.zeros(self.n_metacells)
        
        for k in range(self.n_metacells):
            indices = np.where(labels == k)[0]
            if len(indices) > 1:
                centroids[k] = np.mean(ref_data[indices], axis=0)
                var = np.var(ref_data[indices], axis=0).sum()
                density_scores[k] = 1.0 / (var + 1e-6)
            else:
                density_scores[k] = 0.0
        
        valid_indices = [k for k in range(self.n_metacells) if k not in starving_indices]
        loose_indices = []
        if valid_indices:
            cutoff = np.percentile([density_scores[k] for k in valid_indices], 5)
            loose_indices = [k for k in valid_indices if density_scores[k] <= cutoff]

        composite_scores = sizes * (density_scores ** 0.5)
        n_starving_fixed = 0
        
        sorted_candidates = np.argsort(composite_scores)[::-1]
        candidate_ptr = 0
        
        for poor_idx in starving_indices:
            while candidate_ptr < len(sorted_candidates):
                donor = sorted_candidates[candidate_ptr]
                if donor not in starving_indices and donor != poor_idx:
                    self._split_and_overwrite(donor, poor_idx)
                    sizes[donor] *= 0.5 
                    n_starving_fixed += 1
                    if sizes[donor] < size_threshold * 4: 
                        candidate_ptr += 1
                    break
                candidate_ptr += 1
                
        n_outliers_moved = 0
        for loose_idx in loose_indices:
            indices = np.where(labels == loose_idx)[0]
            if len(indices) < 5: continue 

            dists = np.linalg.norm(ref_data[indices] - centroids[loose_idx], axis=1)
            mean_dist = np.mean(dists)
            std_dist = np.std(dists)
            threshold = mean_dist + 3.0 * std_dist
            
            outlier_local_indices = np.where(dists > threshold)[0]
            if len(outlier_local_indices) > 0:
                outlier_global_indices = indices[outlier_local_indices]
                self._reassign_outliers(outlier_global_indices, centroids)
                n_outliers_moved += len(outlier_global_indices)

        if n_starving_fixed > 0 or n_outliers_moved > 0:
            print(f"  [Respawn] Iter {iteration}: Starving重置={n_starving_fixed}, Loose提纯={len(loose_indices)} (移动离群细胞 {n_outliers_moved} 个)")

    def _split_and_overwrite(self, donor_idx, target_idx):
        donor_weights = self.A[donor_idx, :]
        top_cells = np.argsort(donor_weights)[-2:]
        c_move = top_cells[0] 
        
        self.B[:, target_idx] = 0.0
        self.B[c_move, target_idx] = 1.0 
        
        shared_weight = self.A[donor_idx, :].copy() * 0.5
        self.A[donor_idx, :] = shared_weight
        self.A[target_idx, :] = shared_weight

    def _reassign_outliers(self, cell_indices, all_centroids):
        if self.split_metric == 'pca':
            sub_data = self.views_data[0][cell_indices]
            dists = cdist(sub_data, all_centroids, metric='euclidean')
            nearest_metacells = np.argmin(dists, axis=1)
            
        elif self.split_metric == 'kernel':
            M = self.kernels[0] 
            M_sub = M[cell_indices, :] 
            affinities = M_sub @ self.B
            nearest_metacells = np.argmax(affinities, axis=1)
        else:
            raise ValueError(f"Unknown split_metric: {self.split_metric}")

        for i, cell_idx in enumerate(cell_indices):
            target_k = nearest_metacells[i]
            target_cluster_indices = np.where(self.A.argmax(0) == target_k)[0]
            if len(target_cluster_indices) > 0:
                avg_weight = self.A[target_k, target_cluster_indices].mean()
            else:
                avg_weight = 0.01 
            
            self.A[:, cell_idx] = 0.0 
            self.A[target_k, cell_idx] = avg_weight 

    def _updateA_incremental(self, B, A_prev, weights, iteration):
        n_cells = A_prev.shape[1]
        t1_sum = np.zeros((self.n_metacells, self.n_metacells))
        t2_sum = np.zeros((self.n_metacells, n_cells))
        for idx, M in enumerate(self.kernels):
            w = weights[idx]
            MB = M @ B
            MT_MB = M.T @ MB
            t1_sum += w * (B.T @ MT_MB)
            t2_sum += w * MT_MB.T
        grad_recon = 2.0 * (t1_sum @ A_prev - t2_sum)
        return _numba_update_A_incremental(A_prev, grad_recon, t1_sum, t2_sum, self.lambda_sparse, self.lambda_balance, self.max_franke_wolfe_iters, 10)

    def _updateB_incremental(self, A, B_prev, weights):
        n_cells, n_metacells = B_prev.shape
        B = B_prev
        AA_T = A @ A.T
        G = np.zeros(B.shape) 
        Z_list = [M @ B for M in self.kernels]
        C = np.zeros(B.shape)
        for idx, M in enumerate(self.kernels):
            C += 2.0 * weights[idx] * (M.T @ (M @ A.T))
        
        t = 0
        while t < self.max_franke_wolfe_iters:
            gamma = 2.0 / (t + 2.0)
            if t % 10 == 0:
                G[:] = -C 
                for idx, M in enumerate(self.kernels):
                    term = (M.T @ Z_list[idx]) @ AA_T
                    G += 2.0 * weights[idx] * term
                if self.lambda_consistency > 0 and len(self.kernels) > 1:
                    G += self.lambda_consistency * self._compute_cons_grad_from_Z(Z_list)
            
            min_indices = np.argmin(G, axis=0)
            for idx, M in enumerate(self.kernels):
                M_cols = M[:, min_indices].toarray() if issparse(M) else M[:, min_indices]
                Z_list[idx] = (1.0 - gamma) * Z_list[idx] + gamma * M_cols
            
            B *= (1.0 - gamma)
            for k, idx in enumerate(min_indices): B[idx, k] += gamma
            t += 1
        return B

    def _compute_cons_grad_from_Z(self, Z_list):
        n_views = len(self.kernels)
        grad = np.zeros(Z_list[0].shape)
        count = 0
        for i in range(n_views):
            for j in range(i+1, n_views):
                grad += 2.0 * ((self.kernels[i].T @ Z_list[i] - self.kernels[i].T @ Z_list[j]) + 
                               (self.kernels[j].T @ Z_list[j] - self.kernels[j].T @ Z_list[i]))
                count += 1
        return grad / count if count else grad

    def _compute_loss_efficient(self, kernels, B, A, weights):
        total_loss = 0
        recon_errors = []
        AA_T = A @ A.T 
        for i, (M, w) in enumerate(zip(kernels, weights)):
            term1 = self.kernel_norms_sq[i]
            MB = M @ B
            MT_MB = M.T @ MB
            term2 = np.trace(AA_T @ (B.T @ MT_MB))
            term3 = -2.0 * np.trace(A @ MT_MB)
            err = max(0.0, term1 + term2 + term3)
            recon_errors.append(err)
            total_loss += w * err
            
        sparse_reg = self.lambda_sparse * np.abs(A).sum()
        var_loss = 0.5 * np.sum((A.sum(1) - (A.shape[1]/A.shape[0]))**2)
        bal_reg = self.lambda_balance * var_loss
        
        cons_loss = 0.0
        count = 0
        MB_list = [M @ B for M in kernels]
        for i in range(len(kernels)):
            for j in range(i+1, len(kernels)):
                cons_loss += np.linalg.norm(MB_list[i] - MB_list[j])**2
                count += 1
        cons_reg = self.lambda_consistency * (cons_loss/count if count else 0)
        return total_loss + sparse_reg + bal_reg + cons_reg, np.array(recon_errors), cons_reg, sparse_reg
        
    def _update_weights_consensus(self, kernels, B, old_weights, momentum):
        n_views = len(kernels)
        MB_list = [M @ B for M in kernels]
        avg_distances = np.zeros(n_views)
        for i in range(n_views):
            dists = [np.linalg.norm(MB_list[i] - MB_list[j]) for j in range(n_views) if i != j]
            avg_distances[i] = np.mean(dists) if dists else 0.0
        new_w = 1.0 / (avg_distances**10 + 1e-6)
        new_w /= new_w.sum()
        final_w = momentum * old_weights + (1 - momentum) * new_w
        return final_w / final_w.sum()