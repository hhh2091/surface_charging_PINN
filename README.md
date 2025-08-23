# 表面充电PINN框架

基于PyTorch的物理信息神经网络(PINN)框架，用于解决航天器表面充电问题。该框架采用混合"灰盒"架构，将确定的物理守恒定律作为结构性约束，同时利用神经网络学习物理模型中不确定性最高的部分。

## 🚀 主要特性

- **混合"灰盒"架构**: 结合物理约束和数据驱动学习
- **主网络(VNN)**: 预测表面电位，受数据和物理残差双重约束
- **子网络(NNyield)**: 学习电子激发发射电流密度，替代传统经验公式
- **物理残差约束**: 将瞬态电流平衡方程转化为损失函数
- **多组件损失函数**: 物理残差、边界条件、初始条件和数据损失
- **两阶段优化**: Adam + L-BFGS优化策略
- **自适应权重调整**: 梯度归一化动态平衡各损失项

## 📁 项目结构

```
surface_charging_trae/
├── requirements.txt          # 依赖包列表
├── pinn_model.py            # 核心PINN模型定义
├── loss_functions.py        # 损失函数和自适应权重
├── optimizer.py             # 两阶段优化器
├── data_generator.py        # 数据生成和处理
├── trainer.py               # 训练器和模型管理
├── example_usage.py         # 使用示例
└── README.md               # 项目文档
```

## 🛠️ 安装

1. 克隆或下载项目到本地
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

### 主要依赖

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- tqdm >= 4.62.0

## 🎯 快速开始

### 基本使用

```python
from trainer import PINNTrainer
import torch

# 创建配置
config = {
    'model': {
        'vnn_input_dim': 12,
        'vnn_hidden_dims': [64, 64, 64, 64],
        'yield_input_dim': 4,
        'yield_hidden_dims': [32, 32, 32]
    },
    'data': {
        't_range': [0.0, 10.0],
        'x_range': [-0.1, 0.1],
        'n_collocation': 10000,
        'n_boundary': 1000
    },
    'optimizer': {
        'adam': {'lr': 1e-3, 'max_iter': 5000},
        'lbfgs': {'max_iter': 1000}
    }
}

# 创建训练器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = PINNTrainer(config, device)

# 训练模型
results = trainer.train()

# 保存模型
trainer.save_model('pinn_model.pth')
```

### 运行完整示例

```bash
python example_usage.py
```

## 🧠 模型架构

### 主网络 (VNN)

- **输入**: 12维向量 `[t, x, y, z, ne, Te, ni, Ti, Sflux, α_sun, α_ram, material_id]`
- **输出**: 标量表面电位 `V`
- **架构**: 全连接神经网络 (MLP)
- **激活函数**: Tanh (默认)

### 子网络 (NNyield)

- **输入**: 4维向量 `[V, Je_inc, Mprop]`
- **输出**: 电子激发发射电流密度 `J_emission`
- **架构**: 全连接神经网络
- **激活函数**: ReLU (确保电流密度为正)

## ⚡ 物理模型

### 控制方程

瞬态电流平衡方程：

```
C(x) ∂V(x,t)/∂t = J_net(V, E_env, M_prop, x, t)
```

其中净电流密度：

```
J_net = J_e - J_i - J_se - J_bse - J_ph
```

### 电流分量

1. **等离子体电流** (Je, Ji): 轨道限制理论模型
2. **光电子电流** (Jph): 自限模型
3. **电子激发发射电流** (Jse + Jbse): 神经网络学习

### 物理残差

```
f = C * ∂V_NN/∂t - J_net(V_NN, ...)
```

## 📊 损失函数

总损失函数：

```
L_total = λ_r * L_residual + λ_bc * L_boundary + λ_ic * L_initial + λ_data * L_data
```

- **L_residual**: 物理残差损失 (MSE)
- **L_boundary**: 边界条件损失
- **L_initial**: 初始条件损失
- **L_data**: 数据拟合损失

### 自适应权重调整

支持梯度归一化方法动态调整损失权重，确保各物理约束平衡训练。

## 🔧 优化策略

### 两阶段优化

1. **第一阶段 (Adam)**: 快速收敛到良好区域
   - 学习率: 1e-3
   - 迭代次数: 5000

2. **第二阶段 (L-BFGS)**: 精确收敛到最优解
   - 利用二阶信息
   - 迭代次数: 1000

### 其他优化技术

- 梯度裁剪
- 早停机制
- 学习率调度
- 数据归一化

## 📈 训练监控

框架提供丰富的训练监控功能：

- 实时损失跟踪
- 损失分量可视化
- 权重历史记录
- 学习率变化
- 解的可视化

## 🎛️ 配置选项

### 模型配置

```python
'model': {
    'vnn_input_dim': 12,           # 主网络输入维度
    'vnn_hidden_dims': [64, 64, 64, 64],  # 隐藏层维度
    'vnn_activation': 'tanh',      # 激活函数
    'yield_input_dim': 4,          # 子网络输入维度
    'yield_hidden_dims': [32, 32, 32],    # 子网络隐藏层
    'yield_activation': 'relu'     # 子网络激活函数
}
```

### 数据配置

```python
'data': {
    't_range': [0.0, 10.0],        # 时间范围
    'x_range': [-0.1, 0.1],        # 空间范围
    'ne_range': [1e6, 1e8],        # 电子密度范围
    'Te_range': [1000, 5000],      # 电子温度范围
    'n_collocation': 10000,        # 配置点数量
    'sampling_method': 'latin_hypercube'  # 采样方法
}
```

### 训练配置

```python
'optimizer': {
    'adam': {
        'lr': 1e-3,                # 学习率
        'max_iter': 5000           # 最大迭代次数
    },
    'lbfgs': {
        'max_iter': 1000,          # L-BFGS迭代次数
        'tolerance_grad': 1e-7     # 梯度容忍度
    }
}
```

## 📊 结果分析

### 评估指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **相对误差**: 相对平均绝对误差
- **R²分数**: 决定系数

### 可视化

- 训练历史曲线
- 损失分量演化
- 权重调整历史
- 解的空间分布
- 时间演化曲线

## 🔬 高级功能

### 数据采样方法

- 均匀采样 (uniform)
- 拉丁超立方采样 (latin_hypercube)
- Sobol序列采样 (sobol)

### 归一化方法

- MinMax归一化
- Z-score标准化
- 鲁棒归一化

### 自适应权重策略

- 梯度归一化
- 损失平衡

## 🚨 注意事项

1. **GPU内存**: 大规模训练需要足够的GPU内存
2. **数值稳定性**: 物理参数范围应合理设置
3. **收敛性**: 复杂物理问题可能需要调整网络架构和训练参数
4. **数据质量**: 高质量的训练数据对模型性能至关重要

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个框架！

## 📄 许可证

本项目采用MIT许可证。

## 📚 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

3. Spacecraft charging technology development and validation. NASA Technical Reports.

---

**开发者**: PINN Framework Team  
**版本**: 1.0.0  
**更新日期**: 2024年