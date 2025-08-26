"""PINN模型包

这个包包含了航天器表面充电PINN模型的核心组件：
- physics.py: 物理方程和电流模型
- networks.py: 神经网络架构定义
- dataset.py: 数据集处理和加载
"""

from .physics import *
from .networks import *
from .dataset import *

__version__ = "1.0.0"
__author__ = "PINN Engineering Team"