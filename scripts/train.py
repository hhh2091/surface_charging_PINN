"""训练脚本

航天器表面充电PINN模型的训练主脚本。
实现两阶段优化策略：
1. Adam优化器进行快速收敛
2. L-BFGS优化器进行精确收敛

支持动态损失权重调整和梯度归一化。

Author: PINN Engineering Team
"""

import os
import sys
import yaml
import torch
import numpy as np
import deepxde as dde
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.physics import create_physics_model
from pinn_model.networks import create_networks, print_network_summary
from pinn_model.dataset import create_dataloader


class PINNTrainer:
    """PINN训练器
    
    整合物理模型、神经网络和训练逻辑。
    """
    
    def __init__(self, config: Dict, experiment_name: str = None):
        """初始化训练器
        
        Args:
            config: 配置字典
            experiment_name: 实验名称
        """
        self.config = config
        self.experiment_name = experiment_name or f"pinn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建实验目录
        self.experiment_dir = Path(config['model']['save_dir']) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.physics_model = create_physics_model(config)
        self.networks = create_networks(config)
        self.dataloader = create_dataloader(config)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.loss_history = {'total': [], 'residual': [], 'data': [], 'initial': []}
        
        # 动态权重
        self.loss_weights = {
            'lambda_residual': config['training']['loss_weights']['lambda_residual'],
            'lambda_data': config['training']['loss_weights']['lambda_data'],
            'lambda_initial': config['training']['loss_weights']['lambda_initial']
        }
        
        self.logger.info(f"Initialized PINN trainer for experiment: {self.experiment_name}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.experiment_dir / self.config['logging']['log_file']
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_pinn_problem(self) -> dde.data.PDE:
        """创建PINN问题定义
        
        Returns:
            pinn_problem: DeepXDE PINN问题
        """
        def pde(x, y):
            """物理方程残差函数
            
            Args:
                x: 输入变量 [batch_size, 8] - [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
                y: 网络输出 [batch_size, 1] - [V]
                
            Returns:
                residual: 物理残差
            """
            # 提取输入变量
            t = x[:, 0:1]
            n_e = x[:, 1:2]
            T_e = x[:, 2:3]
            n_i = x[:, 3:4]
            T_i = x[:, 4:5]
            S_flux = x[:, 5:6]
            alpha_sun = x[:, 6:7]
            material_id = x[:, 7:8]
            
            # 表面电位
            V = y
            
            # 计算时间导数
            dV_dt = dde.grad.jacobian(y, x, i=0, j=0)
            
            # 使用子网络预测电子发射电流
            yield_input = torch.cat([V, material_id], dim=1)
            J_emission = self.networks.yield_net(yield_input)
            
            # 计算物理残差
            residual = self.physics_model.charging_ode(
                V, dV_dt, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id, J_emission
            )
            
            return residual
        
        def initial_condition(x):
            """初始条件：t=0时V=0
            
            Args:
                x: 输入变量
                
            Returns:
                initial_value: 初始条件值
            """
            return torch.zeros_like(x[:, 0:1])
        
        # 定义计算域
        data_config = self.config['data_generation']
        
        # 时间域
        t_domain = [data_config['t_start'], data_config['t_end']]
        
        # 参数域
        n_e_domain = data_config['n_e_range']
        T_e_domain = data_config['T_e_range']
        n_i_domain = data_config['n_i_range']
        T_i_domain = data_config['T_i_range']
        S_flux_domain = data_config['S_flux_range']
        alpha_domain = data_config['alpha_sun_range']
        material_domain = [0, len(data_config['materials']) - 1]
        
        # 创建几何域（8维超立方体）
        geom = dde.geometry.Hypercube(
            xmin=[t_domain[0], n_e_domain[0], T_e_domain[0], n_i_domain[0], 
                  T_i_domain[0], S_flux_domain[0], alpha_domain[0], material_domain[0]],
            xmax=[t_domain[1], n_e_domain[1], T_e_domain[1], n_i_domain[1], 
                  T_i_domain[1], S_flux_domain[1], alpha_domain[1], material_domain[1]]
        )
        
        # 创建时间域
        timedomain = dde.geometry.TimeDomain(t_domain[0], t_domain[1])
        
        # 创建时空域
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        
        # 定义初始条件
        ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)
        
        # 创建PDE问题
        pinn_problem = dde.data.PDE(
            geomtime,
            pde,
            [ic],
            num_domain=5000,  # 域内采样点数
            num_boundary=0,   # 边界采样点数（无边界条件）
            num_initial=1000, # 初始条件采样点数
            num_test=1000     # 测试点数
        )
        
        return pinn_problem
    
    def create_deepxde_model(self, pinn_problem) -> dde.Model:
        """创建DeepXDE模型
        
        Args:
            pinn_problem: PINN问题定义
            
        Returns:
            model: DeepXDE模型
        """
        # 创建主网络
        main_config = self.config['network']['main_network']
        net = dde.nn.FNN(
            layer_sizes=main_config['layers'],
            activation=main_config['activation'],
            kernel_initializer="Glorot uniform"
        )
        
        # 创建模型
        model = dde.Model(pinn_problem, net)
        
        return model
    
    def load_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载训练数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            X_data: 输入数据
            y_data: 目标数据
        """
        if not os.path.exists(data_path):
            self.logger.warning(f"Training data not found: {data_path}")
            return None, None
        
        # 加载数据集
        self.dataloader.load_datasets(data_path)
        dataset_info = self.dataloader.get_dataset_info()
        
        self.logger.info(f"Loaded training data: {dataset_info}")
        
        # 获取数据
        train_dataset = self.dataloader.train_dataset
        X_data = train_dataset.inputs.numpy()
        y_data = train_dataset.targets.numpy()
        
        return X_data, y_data
    
    def update_loss_weights(self, losses: Dict[str, float]):
        """动态更新损失权重
        
        Args:
            losses: 当前损失值字典
        """
        if not self.config['training']['use_gradient_normalization']:
            return
        
        # 简单的自适应权重调整策略
        total_loss = sum(losses.values())
        
        if total_loss > 0:
            # 根据损失比例调整权重
            residual_ratio = losses.get('residual', 0) / total_loss
            data_ratio = losses.get('data', 0) / total_loss
            initial_ratio = losses.get('initial', 0) / total_loss
            
            # 平衡各项损失的贡献
            target_ratio = 1.0 / 3.0
            
            if residual_ratio > target_ratio * 1.5:
                self.loss_weights['lambda_residual'] *= 0.95
            elif residual_ratio < target_ratio * 0.5:
                self.loss_weights['lambda_residual'] *= 1.05
            
            if data_ratio > target_ratio * 1.5:
                self.loss_weights['lambda_data'] *= 0.95
            elif data_ratio < target_ratio * 0.5:
                self.loss_weights['lambda_data'] *= 1.05
            
            if initial_ratio > target_ratio * 1.5:
                self.loss_weights['lambda_initial'] *= 0.95
            elif initial_ratio < target_ratio * 0.5:
                self.loss_weights['lambda_initial'] *= 1.05
    
    def train_adam_phase(self, model: dde.Model, X_data: np.ndarray, y_data: np.ndarray):
        """Adam优化阶段
        
        Args:
            model: DeepXDE模型
            X_data: 训练输入数据
            y_data: 训练目标数据
        """
        self.logger.info("Starting Adam optimization phase...")
        
        adam_config = self.config['training']['adam']
        
        # 编译模型
        model.compile(
            optimizer="adam",
            lr=adam_config['learning_rate'],
            loss_weights=[
                self.loss_weights['lambda_residual'],
                self.loss_weights['lambda_data'],
                self.loss_weights['lambda_initial']
            ]
        )
        
        # 添加数据（如果有的话）
        if X_data is not None and y_data is not None:
            model.train(
                iterations=adam_config['iterations'],
                batch_size=self.config['training']['batch_size'],
                display_every=self.config['logging']['print_frequency'],
                callbacks=[
                    dde.callbacks.ModelCheckpoint(
                        str(self.experiment_dir / "adam_checkpoint"),
                        save_better_only=True,
                        period=self.config['model']['save_frequency']
                    )
                ]
            )
        else:
            # 仅使用物理约束训练
            model.train(
                iterations=adam_config['iterations'],
                display_every=self.config['logging']['print_frequency']
            )
        
        self.logger.info("Adam optimization phase completed")
    
    def train_lbfgs_phase(self, model: dde.Model):
        """L-BFGS优化阶段
        
        Args:
            model: DeepXDE模型
        """
        self.logger.info("Starting L-BFGS optimization phase...")
        
        lbfgs_config = self.config['training']['lbfgs']
        
        # 重新编译模型使用L-BFGS
        model.compile(
            optimizer="L-BFGS",
            loss_weights=[
                self.loss_weights['lambda_residual'],
                self.loss_weights['lambda_data'],
                self.loss_weights['lambda_initial']
            ]
        )
        
        # L-BFGS训练
        model.train(
            iterations=lbfgs_config['iterations'],
            display_every=self.config['logging']['print_frequency']
        )
        
        self.logger.info("L-BFGS optimization phase completed")
    
    def save_model(self, model: dde.Model, suffix: str = ""):
        """保存模型
        
        Args:
            model: DeepXDE模型
            suffix: 文件名后缀
        """
        model_path = self.experiment_dir / f"model{suffix}"
        model.save(str(model_path))
        
        # 保存网络状态
        networks_path = self.experiment_dir / f"networks{suffix}.pth"
        torch.save({
            'main_net_state_dict': self.networks.main_net.state_dict(),
            'yield_net_state_dict': self.networks.yield_net.state_dict(),
            'loss_weights': self.loss_weights,
            'config': self.config
        }, networks_path)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def train(self, data_path: str = None):
        """完整训练流程
        
        Args:
            data_path: 训练数据路径（可选）
        """
        self.logger.info("Starting PINN training...")
        
        # 打印网络架构
        print_network_summary(self.networks)
        
        # 加载训练数据
        X_data, y_data = None, None
        if data_path:
            X_data, y_data = self.load_training_data(data_path)
        
        # 创建PINN问题
        pinn_problem = self.create_pinn_problem()
        
        # 添加观测数据（如果有）
        if X_data is not None and y_data is not None:
            observe_data = dde.icbc.PointSetBC(X_data, y_data)
            pinn_problem.add_bcs([observe_data])
        
        # 创建模型
        model = self.create_deepxde_model(pinn_problem)
        
        try:
            # 第一阶段：Adam优化
            self.train_adam_phase(model, X_data, y_data)
            self.save_model(model, "_adam")
            
            # 第二阶段：L-BFGS优化
            self.train_lbfgs_phase(model)
            self.save_model(model, "_final")
            
            self.logger.info("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        return model


def load_config(config_path: str) -> Dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train spacecraft charging PINN model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--data', type=str, default=None,
                       help='Training data file path (optional)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置DeepXDE后端
    dde.config.set_default_float('float32')
    
    # 创建训练器
    trainer = PINNTrainer(config, args.experiment)
    
    # 开始训练
    try:
        model = trainer.train(args.data)
        print("\nTraining completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model(model, "_interrupted")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()