import scanpy as sc
from stark.core.hdata import HData

def create_hdata_from_adata(adata, 
                            data_dir=None, 
                            output_dir=None, 
                            genome_reference_path=None, 
                            chrom_list=None, 
                            resolution=None):
    """
    从 AnnData 对象创建 HData 对象。
    这个函数将 AnnData 中的 obs 和 embedding 数据提取出来，存储到 HData 中。
    """
    
    if adata is None:
        raise ValueError("输入的 AnnData 对象不能为空")
    if adata.obs is None or adata.obs.empty:
        raise ValueError("AnnData 对象的 obs 不能为空")
    if data_dir is None:
        raise ValueError("数据目录 (data_dir) 不能为空")
    if output_dir is None:
        raise ValueError("输出目录 (output_dir) 不能为空")
    if genome_reference_path is None:
        raise ValueError("基因组参考路径 (genome_reference_path) 不能为空")
    if chrom_list is None or not chrom_list:
        raise ValueError("染色体列表 (chrom_list) 不能为空")
    if resolution is None or not resolution:
        raise ValueError("分辨率列表 (resolutions) 不能为空")
    
    hdata = HData(
        data_dir=data_dir,
        output_dir=output_dir,
        genome_reference_path=genome_reference_path,
        chrom_list=chrom_list,
        resolutions=resolution
    )
    
    hdata.views_pca[resolution] = adata.uns.get('X_pca', None)
    hdata.views_umap[resolution] = adata.uns.get('X_umap', None)
    
    hdata.obs = adata.obs.copy()

    
    return hdata
    