# 航天器表面充电PINN预测系统

一个基于物理信息神经网络(PINN)的航天器表面充电预测工程项目，采用参数化灰箱方法，结合已知物理方程和可学习的神经网络组件。

## 项目概述

本项目实现了一个用于预测航天器表面充电现象的PINN模型，该模型：

- **灰箱方法**：结合已知的物理方程（电子/离子/光电子电流）和可学习的神经网络（电子发射电流）
- **参数化输入**：支持多种环境参数和材料类型
- **两阶段优化**：Adam + L-BFGS优化策略确保高精度收敛
- **物理约束**：通过物理残差损失确保预测结果符合物理定律
- **全面评估**：包含物理核查和可视化分析

## 项目结构

```
spacecraft_charging_pinn/
├── configs/
│   └── default_config.yaml         # 集中管理所有超参数和配置
├── data_generation/
│   └── generate_data.py            # 模拟生成高保真训练/测试数据
├── pinn_model/
│   ├── __init__.py
│   ├── physics.py                  # 定义所有已知的物理方程（电流模型）
│   ├── networks.py                 # 定义主网络和子网络的PyTorch架构
│   └── dataset.py                  # 定义处理参数化输入的数据加载器
├── scripts/
│   ├── train.py                    # 训练主脚本
│   ├── evaluate.py                 # 评估模型性能并进行物理核查的脚本
│   └── visualize.py                # 结果可视化脚本
├── README.md                       # 项目说明、安装和使用指南
└── requirements.txt                # 项目依赖库
```

## 核心物理模型

### 控制方程

表面充电过程由以下瞬态电流平衡常微分方程描述：

```
C * dV/dt = J_net * A
```

其中：
- `C`: 等效电容 (常数)
- `A`: 表面积 (常数) 
- `V`: 表面电位
- `J_net`: 净电流密度

### 净电流密度

```
J_net = J_e - J_i - J_ph - J_emission
```

- `J_e`: 电子电流（轨道限制理论）
- `J_i`: 离子电流（轨道限制理论）
- `J_ph`: 光电子电流（饱和模型）
- `J_emission`: 电子发射电流（神经网络建模）

### 模型输入输出

**主网络输入**：
- 时间: `t`
- 环境参数: `n_e`, `T_e`, `n_i`, `T_i`
- 光照参数: `S_flux`, `alpha_sun`
- 材料标识: `material_id`

**主网络输出**：
- 表面电位: `V(t, params)`

**子网络输入**：
- 表面电位: `V`
- 材料标识: `material_id`

**子网络输出**：
- 电子发射电流密度: `J_emission`

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd spacecraft_charging_pinn
```

2. **创建虚拟环境**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **验证安装**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 使用指南

### 1. 配置实验

编辑 `configs/default_config.yaml` 文件来定义实验参数：

```yaml
# 数据生成参数
data_generation:
  n_scenarios_train: 100
  n_scenarios_test: 50
  t_end: 7200  # 2小时
  
# 网络架构
network:
  main_network:
    hidden_layers: [64, 64, 64]
    activation: 'tanh'
  yield_network:
    hidden_layers: [32, 32]
    activation: 'tanh'

# 训练参数
training:
  adam_epochs: 10000
  lbfgs_epochs: 1000
  learning_rate: 0.001
  batch_size: 256
```

### 2. 生成训练数据

```bash
# 使用默认配置生成数据
python data_generation/generate_data.py

# 使用自定义配置
python data_generation/generate_data.py --config configs/custom_config.yaml --output_dir data/custom_experiment

# 指定求解方法
python data_generation/generate_data.py --method ode_solver
# 可选方法: ode_solver, forward_euler, analytical_approx, auto
```

### 3. 训练模型

```bash
# 基本训练
python scripts/train.py --config configs/default_config.yaml --data_dir data/generated_data

# 指定输出目录
python scripts/train.py --config configs/default_config.yaml --data_dir data/generated_data --output_dir results/experiment_1

# 从检查点继续训练
python scripts/train.py --config configs/default_config.yaml --data_dir data/generated_data --resume results/experiment_1/checkpoint_epoch_5000.pth
```

### 4. 评估模型

```bash
# 基本评估
python scripts/evaluate.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --test_data data/generated_data/test_data.csv

# 详细物理核查
python scripts/evaluate.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --test_data data/generated_data/test_data.csv --physics_check

# 指定输出目录
python scripts/evaluate.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --test_data data/generated_data/test_data.csv --output_dir evaluation_results
```

### 5. 结果可视化

```bash
# 基本可视化
python scripts/visualize.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth

# 使用自定义场景
python scripts/visualize.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --scenarios data/custom_scenarios.csv

# 指定输出目录
python scripts/visualize.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --output_dir visualization_results
```

## 完整工作流程示例

```bash
# 1. 生成训练数据
python data_generation/generate_data.py --config configs/default_config.yaml --n_train 200 --n_test 100 --output_dir data/experiment_1

# 2. 训练模型
python scripts/train.py --config configs/default_config.yaml --data_dir data/experiment_1 --output_dir results/experiment_1

# 3. 评估模型
python scripts/evaluate.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --test_data data/experiment_1/test_data.csv --output_dir evaluation/experiment_1

# 4. 可视化结果
python scripts/visualize.py --config configs/default_config.yaml --model results/experiment_1/final_model.pth --output_dir visualization/experiment_1
```

## 高级功能

### 自定义物理模型

在 `pinn_model/physics.py` 中修改物理方程：

```python
def custom_electron_current(self, params: Dict[str, float]) -> float:
    """自定义电子电流模型"""
    # 实现自定义物理方程
    pass
```

### 自定义网络架构

在 `pinn_model/networks.py` 中修改网络结构：

```python
class CustomMainNetwork(nn.Module):
    """自定义主网络架构"""
    def __init__(self, config):
        # 实现自定义网络结构
        pass
```

### 实验跟踪

项目支持Weights & Biases实验跟踪：

```bash
# 安装wandb
pip install wandb

# 登录wandb
wandb login

# 启用wandb跟踪
python scripts/train.py --config configs/default_config.yaml --data_dir data/experiment_1 --use_wandb --project spacecraft_charging
```

## 物理核查说明

模型评估包含以下物理合理性检查：

1. **阴影区转换**：验证进入地球阴影区时电位快速下降
2. **平静空间天气**：验证光照区稳定充电至微弱正值
3. **材料对比**：验证不同材料的充电行为差异
4. **物理残差**：检查模型是否满足物理方程
5. **初始条件**：验证初始时刻的边界条件

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批大小：修改配置文件中的 `batch_size`
   - 使用CPU：设置 `device.use_gpu: false`

2. **训练不收敛**
   - 调整学习率：减小 `learning_rate`
   - 增加训练轮数：增大 `adam_epochs`
   - 检查损失权重：调整 `lambda_residual`, `lambda_data`

3. **物理核查失败**
   - 检查物理参数范围
   - 增加物理约束权重
   - 验证训练数据质量

### 日志分析

训练和评估过程会生成详细日志：

- `training.log`: 训练过程日志
- `evaluation.log`: 评估过程日志
- `visualization.log`: 可视化过程日志

## 性能优化

### GPU加速

```yaml
# 在配置文件中启用GPU
device:
  use_gpu: true
  gpu_id: 0
```

### 并行数据加载

```yaml
# 在配置文件中设置数据加载器
dataloader:
  num_workers: 4
  pin_memory: true
```

### 混合精度训练

```yaml
# 在配置文件中启用混合精度
training:
  use_amp: true
```

## 扩展开发

### 添加新材料

1. 在配置文件中添加材料参数
2. 更新 `material_id` 映射
3. 修改子网络以支持新材料

### 添加新物理效应

1. 在 `physics.py` 中实现新的电流模型
2. 更新净电流计算
3. 调整损失函数权重

### 集成外部仿真器

1. 创建数据接口模块
2. 实现数据格式转换
3. 更新数据加载器

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{spacecraft_charging_pinn,
  title={Spacecraft Surface Charging PINN Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/spacecraft-charging-pinn}
}
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 支持

如有问题或建议，请：

1. 查看 [FAQ](docs/FAQ.md)
2. 搜索 [Issues](https://github.com/your-repo/spacecraft-charging-pinn/issues)
3. 创建新的Issue
4. 联系维护者：your.email@example.com

---

**注意**：本项目仅用于研究和教育目的。在实际航天任务中使用前，请进行充分的验证和测试。