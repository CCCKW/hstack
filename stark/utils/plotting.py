import numpy as np
import matplotlib.pyplot as plt

class PlottingMixin:
    """提供可视化功能的 Mixin 类"""
    
    def plot_initialization(self, umap_coords, title="Initialization Waypoints"):
        if not getattr(self, 'initialized', False): 
            raise RuntimeError("请先运行 initialize")
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
        wp_coords = umap_coords[self.waypoints, :]
        plt.scatter(wp_coords[:, 0], wp_coords[:, 1], c='red', s=60, edgecolors='black', linewidth=1, label='Initial Waypoints', zorder=10)
        plt.title(title)
        plt.axis('off')
        plt.legend()
        plt.show()

    def plot_specific_metacell(self, umap_coords, metacell_id):
        labels = self.A.argmax(axis=0)
        indices = np.where(labels == metacell_id)[0]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.3, label='Background')
        
        if len(indices) > 0:
            target_coords = umap_coords[indices]
            plt.scatter(target_coords[:, 0], target_coords[:, 1], c='red', s=20, label=f'Metacell {metacell_id}')
            center = np.mean(target_coords, axis=0)
            plt.scatter(center[0], center[1], c='black', marker='x', s=100, linewidth=2, label='Centroid')
            
        plt.title(f"Visual Diagnosis: Metacell {metacell_id}")
        plt.legend()
        plt.show()

    def plot_metacells(self, umap_coords, title="Final Metacell Positions", min_size=50, max_size=500, show_idx=False):
        self.labels = self.A.argmax(axis=0)
        metacell_coords = []
        metacell_counts = []
        present_indices = np.unique(self.labels)
        
        for k in present_indices:
            indices = np.where(self.labels == k)[0]
            metacell_coords.append(np.mean(umap_coords[indices], axis=0))
            metacell_counts.append(len(indices))
        
        metacell_coords = np.array(metacell_coords)
        metacell_counts = np.array(metacell_counts)
        
        if len(metacell_counts) == 0:
            print("警告: 没有发现活跃的 metacells。")
            return

        if len(metacell_counts) > 1 and metacell_counts.max() > metacell_counts.min():
            norm_sizes = (metacell_counts - metacell_counts.min()) / (metacell_counts.max() - metacell_counts.min())
            plot_sizes = min_size + norm_sizes * (max_size - min_size)
        else:
            plot_sizes = np.full(len(metacell_counts), (min_size + max_size) / 2)

        plt.figure(figsize=(10, 8))
        plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgrey', s=5, alpha=0.5, rasterized=True)
        plt.scatter(metacell_coords[:, 0], metacell_coords[:, 1], 
                    c='blue', s=plot_sizes, edgecolors='white', linewidth=1, alpha=0.8, zorder=10)
        
        min_c = metacell_counts.min()
        max_c = metacell_counts.max()
        mid_c = int((min_c + max_c) / 2)
        legend_sizes = [min_size, (min_size+max_size)/2, max_size]
        legend_labels = [f'{min_c} cells', f'{mid_c} cells', f'{max_c} cells']
        
        handles = []
        for s, l in zip(legend_sizes, legend_labels):
            handles.append(plt.scatter([], [], c='blue', alpha=0.8, s=s, edgecolors='white', label=l))
        handles.append(plt.scatter([], [], c='lightgrey', s=20, label='Single Cells'))
        
        plt.legend(handles=handles, title="Metacell Size (Count)", loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.5, borderpad=1)
        
        if show_idx:
            for i, k in enumerate(present_indices):
                plt.text(metacell_coords[i, 0], metacell_coords[i, 1], str(k), 
                         fontsize=10, ha='center', va='center', color='black', fontweight='bold', zorder=20)
        
        plt.title(f"{title}\n(Metacells: {len(metacell_coords)}, Count Range: {min_c}-{max_c})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()