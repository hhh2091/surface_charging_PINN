# 航天器表面充电PINN预测系统

## 项目简介

本项目实现了一个基于参数化灰箱物理信息神经网络（Parametric Grey-Box PINN）的航天器表面充电预测系统。该系统结合了已知的物理方程和可学习的神经网络组件，用于预测不同环境条件下航天器表面的充电电位。

### 核心特性

- **灰箱PINN架构**：结合已知物理方程（电子/离子/光电子电流）和可学习组件（电子发射电流）
- **参数化设计**：支持多种环境参数和材料类型
- **两阶段优化**：Adam + L-BFGS优化策略确保收敛精度
- **物理约束**：基于瞬态电流平衡ODE的物理残差
- **全面评估**：包含精度评估和物理合理性核查

## 项目结构

```
spacecraft_charging_pinn/
├── configs/
│   └── default_config.yaml         # 配置文件
├── data_generation/
│   ├── __init__.py
│   └── generate_data.py            # 数据生成脚本
├── pinn_model/
│   ├── __init__.py
│   ├── physics.py                  # 物理方程模块
│   ├── networks.py                 # 神经网络架构
│   └── dataset.py                  # 数据处理模块
├── scripts/
│   ├── __init__.py
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── visualize.py                # 可视化脚本
├── README.md                       # 项目说明
└── requirements.txt                # 依赖列表
```

## 安装指南

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 克隆项目

```bash
git clone <repository_url>
cd spacecraft_charging_pinn
```

### 3. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv pinn_env
source pinn_env/bin/activate  # Linux/Mac
# 或
pinn_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; import deepxde; print('Installation successful!')"
```

## 快速开始

### 1. 生成训练数据

```bash
# 使用默认配置生成数据
python data_generation/generate_data.py --config configs/default_config.yaml --output data/

# 自定义参数
python data_generation/generate_data.py --config configs/default_config.yaml --output data/ --num_samples 10000
```

### 2. 训练模型

```bash
# 基础训练（仅使用物理约束）
python scripts/train.py --config configs/default_config.yaml

# 使用生成的数据训练
python scripts/train.py --config configs/default_config.yaml --data data/spacecraft_charging_dataset.csv

# 自定义实验名称
python scripts/train.py --config configs/default_config.yaml --experiment my_experiment
```

### 3. 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate.py --config configs/default_config.yaml --model experiments/my_experiment/model_final.ckpt

# 使用测试数据评估
python scripts/evaluate.py --config configs/default_config.yaml --model experiments/my_experiment/model_final.ckpt --test_data data/test_data.csv
```

### 4. 结果可视化

```bash
# 可视化训练结果
python scripts/visualize.py --config configs/default_config.yaml --model experiments/my_experiment/model_final.ckpt
```

## 配置说明

主要配置文件为 `configs/default_config.yaml`，包含以下部分：

### 数据生成配置

```yaml
data_generation:
  # 环境参数范围
  n_e_range: [1e5, 1e7]      # 电子密度 (m^-3)
  T_e_range: [0.1, 5.0]      # 电子温度 (eV)
  n_i_range: [1e5, 1e7]      # 离子密度 (m^-3)
  T_i_range: [0.01, 1.0]     # 离子温度 (eV)
  
  # 光照参数
  S_flux_range: [0, 1500]    # 太阳辐射通量 (W/m^2)
  alpha_sun_range: [0, 90]   # 入射角 (度)
  
  # 材料类型
  materials: ['Kapton', 'Aluminum', 'Teflon', 'Carbon']
```

### 网络架构配置

```yaml
network:
  main_network:
    layers: [8, 64, 64, 64, 1]  # 主网络层数
    activation: 'tanh'          # 激活函数
  
  yield_network:
    layers: [2, 32, 32, 1]      # 子网络层数
    activation: 'swish'         # 激活函数
```

### 训练配置

```yaml
training:
  # Adam优化器
  adam:
    learning_rate: 0.001
    iterations: 20000
  
  # L-BFGS优化器
  lbfgs:
    iterations: 5000
  
  # 损失权重
  loss_weights:
    lambda_residual: 1.0
    lambda_data: 100.0
    lambda_initial: 1.0
```

## 物理模型说明

### 控制方程

系统基于瞬态电流平衡常微分方程：

```
C * dV/dt = J_net * A
```

其中：
- `C`: 等效电容
- `V`: 表面电位
- `A`: 表面积
- `J_net`: 净电流密度

### 电流模型

净电流密度包含四个组成部分：

```
J_net = J_e - J_i - J_ph - J_emission
```

1. **电子电流 (J_e)**：基于轨道限制理论
2. **离子电流 (J_i)**：基于轨道限制理论
3. **光电子电流 (J_ph)**：受电位抑制的饱和模型
4. **电子发射电流 (J_emission)**：由神经网络学习

### 网络架构

- **主网络 (V_NN)**：输入环境参数，输出表面电位
- **子网络 (NN_yield)**：输入电位和材料ID，输出电子发射电流

## 使用示例

### 自定义实验

1. **修改配置文件**：
   ```bash
   cp configs/default_config.yaml configs/my_config.yaml
   # 编辑 my_config.yaml
   ```

2. **运行实验**：
   ```bash
   python scripts/train.py --config configs/my_config.yaml --experiment custom_exp
   ```

3. **评估结果**：
   ```bash
   python scripts/evaluate.py --config configs/my_config.yaml --model experiments/custom_exp/model_final.ckpt
   ```

### 批量实验

```bash
# 创建批量实验脚本
for lr in 0.001 0.0001; do
    for layers in "[8,32,32,1]" "[8,64,64,1]"; do
        python scripts/train.py --config configs/default_config.yaml --experiment "exp_lr${lr}_${layers}"
    done
done
```

## 评估指标

### 精度指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **相对误差**: 平均相对误差

### 物理核查

1. **地球阴影区测试**：验证无光照时电位下降
2. **平静光照区测试**：验证稳定条件下的电位范围
3. **材料对比测试**：验证不同材料的充电差异
4. **时间演化测试**：验证充电过程的时间稳定性

## 故障排除

### 常见问题

1. **CUDA内存不足**：
   - 减小批大小
   - 使用CPU训练
   - 减少网络层数

2. **训练不收敛**：
   - 调整学习率
   - 增加训练迭代次数
   - 检查损失权重平衡

3. **物理核查失败**：
   - 检查物理参数设置
   - 验证网络架构
   - 增加物理约束权重

### 调试技巧

```bash
# 启用详细日志
export DEEPXDE_VERBOSE=1

# 使用小数据集测试
python data_generation/generate_data.py --num_samples 100
python scripts/train.py --config configs/default_config.yaml --data data/small_dataset.csv
```

## 扩展开发

### 添加新材料

1. 在配置文件中添加材料名称
2. 更新物理模型参数
3. 重新生成数据和训练

### 自定义物理方程

1. 修改 `pinn_model/physics.py`
2. 更新相应的配置参数
3. 验证物理合理性

### 新的网络架构

1. 在 `pinn_model/networks.py` 中定义新网络
2. 更新配置文件
3. 测试收敛性能

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{spacecraft_charging_pinn,
  title={Spacecraft Charging PINN Prediction System},
  author={PINN Engineering Team},
  year={2024},
  url={<repository_url>}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 项目维护者：PINN Engineering Team
- 邮箱：[your-email@example.com]
- 问题反馈：[GitHub Issues]

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 实现基础PINN架构
- 支持多材料预测
- 完整的评估体系