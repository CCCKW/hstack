"""
Stark: Single-cell multi-view Hi-C data analysis pipeline.
"""

# 1. 暴露核心数据容器 (对标 Scanpy 的 AnnData)
from .core.hdata import HData
from .core.create_hdata import create_hdata_from_adata

# 2. 暴露三大核心操作模块
from . import pp
from . import tl
from . import pl

# 3. 设置版本号 (可选)
__version__ = "0.1.0"