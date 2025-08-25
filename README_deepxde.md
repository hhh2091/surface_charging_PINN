# DeepXDE-based Physics-Informed Neural Network for Spacecraft Surface Charging

基于DeepXDE框架的航天器表面充电物理信息神经网络(PINN)实现。该框架采用混合"灰盒"架构，结合确定的物理守恒定律和数据驱动的神经网络学习。

## 🚀 主要特性

### 核心架构
- **混合"灰盒"设计**: 主网络VNN预测表面电位，子网络NNyield学习电子激发发射电流
- **物理约束集成**: 基于瞬态电流平衡方程的PDE残差
- **多电流组件建模**: 等离子体电流、光电子电流、电子激发发射电流
- **DeepXDE框架**: 利用DeepXDE的高效PDE求解能力

### 技术特性
- **自适应训练**: 动态损失权重调整和梯度归一化
- **两阶段优化**: Adam + L-BFGS优化策略
- **多种边界条件**: 支持Dirichlet、Neumann、周期性边界条件
- **数据融合**: 结合合成数据和实验数据约束
- **可扩展架构**: 支持多种材料和环境参数

## 📁 项目结构

```
deepxde_pinn/
├── deepxde_pinn_model.py      # 核心PINN模型定义
├── boundary_conditions.py     # 边界条件和初始条件管理
├── deepxde_loss_functions.py  # 损失函数和自适应权重
├── deepxde_optimizer.py       # 两阶段优化器实现
├── deepxde_trainer.py         # 完整训练系统
├── deepxde_example_usage.py   # 使用示例
├── requirements.txt           # 依赖包列表
└── README_deepxde.md         # 本文档
```

## 🛠️ 安装

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- DeepXDE 1.10+

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd deepxde_pinn
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```python
import deepxde as dde
import torch
print(f"DeepXDE version: {dde.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

## 🚀 快速开始

### 基本使用

```python
from deepxde_trainer import DeepXDEPINNTrainer

# 创建配置
config = {
    'model': {
        'domain_bounds': {
            't': [0.0, 1.0],
            'x': [0.0, 1.0],
            # ... 其他维度
        },
        'physical_params': {
            'C': 1e-9,  # 电容
            'q_e': -1.602e-19,  # 电子电荷
            # ... 其他物理参数
        },
        'network_config': {
            'hidden_layers': [64, 64, 64, 64],
            'activation': 'tanh'
        }
    },
    # ... 其他配置
}

# 创建训练器
trainer = DeepXDEPINNTrainer(config)

# 训练模型
results = trainer.train()

# 进行预测
X_test = # ... 测试数据
predictions = trainer.predict(X_test)
```

### 运行完整示例

```bash
python deepxde_example_usage.py
```

## 🧠 模型架构

### 主网络 (VNN)
- **输入**: 12维向量 [t, x, y, z, n_e, T_e, n_i, T_i, S_flux, α_sun, α_ram, material_id]
- **输出**: 标量表面电位 V
- **架构**: 全连接神经网络 (MLP)
- **激活函数**: tanh (默认)

### 子网络 (NNyield)
- **功能**: 学习电子激发发射电流密度
- **输入**: [V, J_e_inc, material_properties]
- **输出**: 总电子激发发射电流 J_emission
- **优势**: 替代复杂的经验公式

## ⚡ 物理模型

### 控制方程
瞬态电流平衡方程:
```
C(x) ∂V/∂t = J_net(V, E_env, M_prop, x, t)
```

### 电流组件

1. **等离子体电流** (J_e, J_i)
   - 轨道限制理论模型
   - 吸引性电位: `J_s(V) = J_s0(1 - q_s*V/(k_B*T_s))`
   - 排斥性电位: `J_s(V) = J_s0*exp(q_s*V/(k_B*T_s))`

2. **光电子电流** (J_ph)
   - 自限模型
   - V ≤ 0: `J_ph(V) = J_ph0`
   - V > 0: `J_ph(V) = J_ph0*exp(-V/V_ph)`

3. **电子激发发射电流** (J_se + J_bse)
   - 神经网络建模: `J_emission = NNyield(V, J_e_inc, M_prop)`

## 📊 损失函数

### 多组件损失
```
L_total = λ_r*L_residual + λ_bc*L_boundary + λ_ic*L_initial + λ_data*L_data
```

- **L_residual**: PDE残差损失
- **L_boundary**: 边界条件损失
- **L_initial**: 初始条件损失
- **L_data**: 数据约束损失

### 自适应权重调整
- **梯度归一化**: 根据梯度动态调整权重
- **损失平衡**: 平衡不同损失项的贡献
- **退火策略**: 训练过程中逐步调整权重

## 🔧 优化策略

### 两阶段优化

1. **第一阶段 - Adam优化器**
   - 快速收敛到良好区域
   - 学习率: 1e-3 (默认)
   - 迭代次数: 10,000 (默认)

2. **第二阶段 - L-BFGS优化器**
   - 精确收敛到最优解
   - 利用二阶信息
   - 最大迭代: 50,000 (默认)

### 学习率调度
- **指数衰减**: γ^(epoch/step_size)
- **步长衰减**: 固定步长降低学习率
- **余弦退火**: 余弦函数调度

## 📈 训练监控

### 损失跟踪
- 实时损失监控
- 各组件损失分析
- 权重演化可视化

### 早停机制
- 基于验证损失的早停
- 可配置的耐心参数
- 最佳权重恢复

### 可视化
- 训练曲线绘制
- 解的时空演化
- 残差分布分析

## ⚙️ 配置选项

### 模型配置
```python
'model': {
    'domain_bounds': {...},      # 定义域边界
    'physical_params': {...},    # 物理参数
    'network_config': {...}      # 网络结构
}
```

### 边界条件配置
```python
'boundary_conditions': {
    'grounded_surface': False,   # 接地表面
    'floating_potential': -5.0,  # 浮动电位
    'initial_conditions': {...}  # 初始条件
}
```

### 训练配置
```python
'training': {
    'num_domain_points': 10000,  # 域内配置点数量
    'num_boundary_points': 2000, # 边界点数量
    'batch_size': None,          # 批大小
    'seed': 42                   # 随机种子
}
```

## 📊 结果分析

### 评估指标
- **残差统计**: 均值、标准差、最大值、RMS
- **收敛性**: 损失收敛曲线
- **物理一致性**: PDE残差分布

### 可视化工具
- 解的时空演化动画
- 等势线图
- 电流密度分布
- 训练历史曲线

## 🔬 高级功能

### 多材料支持
```python
'material_properties': {
    0: {'work_function': 4.5, 'secondary_yield': 1.2},  # 铝
    1: {'work_function': 5.1, 'secondary_yield': 0.8},  # 钛
    # ... 更多材料
}
```

### 数据融合
- 合成数据生成
- 实验数据加载
- 多源数据约束

### 不确定性量化
- 贝叶斯神经网络扩展
- 预测区间估计
- 敏感性分析

## 🚨 重要注意事项

### 数值稳定性
- 输入数据归一化
- 梯度裁剪
- 权重初始化策略

### 计算效率
- GPU加速支持
- 批处理优化
- 内存管理

### 物理约束
- 确保电荷守恒
- 验证边界条件
- 检查初始条件一致性

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📚 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.

3. Garrett, H. B., & Whittlesey, A. C. (2012). *Spacecraft charging, an update*. IEEE Transactions on Plasma Science, 40(2), 230-250.

4. Hastings, D., & Garrett, H. (2004). *Spacecraft-Environment Interactions*. Cambridge University Press.

## 📞 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [https://github.com/yourusername/deepxde-pinn-surface-charging]

---

**注意**: 这是一个研究级别的实现，用于学术和研究目的。在生产环境中使用前，请进行充分的验证和测试。