# PINN模型包初始化文件
"""
航天器表面充电PINN模型包

包含以下模块:
- physics: 物理方程和电流模型
- networks: 神经网络架构定义
- dataset: 数据集处理和加载
"""

__version__ = "1.0.0"
__author__ = "PINN Team"

from .physics import *
from .networks import *
from .dataset import *