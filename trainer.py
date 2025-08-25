import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from pinn_model import SurfaceChargingPINN
from loss_functions import PINNLoss, AdaptiveWeighting
from optimizer import TwoStageOptimizer, LearningRateScheduler, EarlyStopping
from data_generator import DataGenerator, DataNormalizer

class PINNTrainer:
    """PINN训练器"""
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Args:
            config: 训练配置
            device: 计算设备
        """
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = SurfaceChargingPINN(config['model']).to(self.device)
        
        # 创建损失函数
        loss_weights = config.get('loss_weights', {
            'residual': 1.0, 'boundary': 1.0, 'initial': 1.0, 'data': 1.0
        })
        self.loss_fn = PINNLoss(loss_weights)
        
        # 自适应权重调整
        adaptive_config = config.get('adaptive_weighting', {'method': 'gradient_normalization'})
        self.adaptive_weighting = AdaptiveWeighting(**adaptive_config)
        
        # 创建优化器
        self.optimizer = TwoStageOptimizer(self.model, config['optimizer'])
        
        # 创建数据生成器
        self.data_generator = DataGenerator(config['data'], self.device)
        
        # 数据归一化器
        self.input_normalizer = DataNormalizer(config.get('input_normalization', 'minmax'))
        self.output_normalizer = DataNormalizer(config.get('output_normalization', 'zscore'))
        
        # 早停机制
        early_stop_config = config.get('early_stopping', {'patience': 200})
        self.early_stopping = EarlyStopping(**early_stop_config)
        
        # 训练历史
        self.training_history = {
            'total_losses': [],
            'loss_components': {'residual': [], 'boundary': [], 'initial': [], 'data': []},
            'weights_history': [],
            'learning_rates': [],
            'epochs': 0
        }
        
        # 保存路径
        self.save_dir = config.get('save_dir', './results')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"PINN Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def prepare_data(self) -> Dict[str, torch.Tensor]:
        """准备训练数据"""
        data_config = self.config['data']
        
        print("Generating training data...")
        
        # 生成各种数据点
        collocation_points = self.data_generator.generate_collocation_points(
            data_config.get('n_collocation', 10000),
            data_config.get('sampling_method', 'uniform')
        )
        
        boundary_points, boundary_values = self.data_generator.generate_boundary_conditions(
            data_config.get('n_boundary', 1000)
        )
        
        initial_points, initial_values = self.data_generator.generate_initial_conditions(
            data_config.get('n_initial', 1000)
        )
        
        data_points, data_values = self.data_generator.generate_synthetic_data(
            data_config.get('n_data', 2000),
            data_config.get('noise_level', 0.01)
        )
        
        print(f"Generated data shapes:")
        print(f"  Collocation points: {collocation_points.shape}")
        print(f"  Boundary points: {boundary_points.shape}, values: {boundary_values.shape}")
        print(f"  Initial points: {initial_points.shape}, values: {initial_values.shape}")
        print(f"  Data points: {data_points.shape}, values: {data_values.shape}")
        
        # 检查数据中的NaN值
        all_data = {
            'collocation_points': collocation_points,
            'boundary_points': boundary_points,
            'boundary_values': boundary_values,
            'initial_points': initial_points,
            'initial_values': initial_values,
            'data_points': data_points,
            'data_values': data_values
        }
        
        for name, tensor in all_data.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: NaN or Inf detected in {name}")
                all_data[name] = torch.where(torch.isnan(tensor) | torch.isinf(tensor),
                                           torch.tensor(0.0, device=tensor.device), tensor)
        
        collocation_points = all_data['collocation_points']
        boundary_points = all_data['boundary_points']
        boundary_values = all_data['boundary_values']
        initial_points = all_data['initial_points']
        initial_values = all_data['initial_values']
        data_points = all_data['data_points']
        data_values = all_data['data_values']
        
        # 数据归一化
        if self.config.get('normalize_inputs', True):
            # 拟合输入归一化器
            all_input_points = torch.cat([collocation_points, boundary_points, initial_points, data_points], dim=0)
            print(f"Fitting input normalizer with {all_input_points.shape[0]} points")
            self.input_normalizer.fit(all_input_points)
            
            # 归一化输入
            collocation_points = self.input_normalizer.transform(collocation_points)
            boundary_points = self.input_normalizer.transform(boundary_points)
            initial_points = self.input_normalizer.transform(initial_points)
            data_points = self.input_normalizer.transform(data_points)
        
        if self.config.get('normalize_outputs', True):
            # 拟合输出归一化器
            all_output_values = torch.cat([boundary_values, initial_values, data_values], dim=0)
            print(f"Fitting output normalizer with {all_output_values.shape[0]} values")
            self.output_normalizer.fit(all_output_values)
            
            # 归一化输出
            boundary_values = self.output_normalizer.transform(boundary_values)
            initial_values = self.output_normalizer.transform(initial_values)
            data_values = self.output_normalizer.transform(data_values)
        
        data_dict = {
            'collocation_points': collocation_points,
            'boundary_points': boundary_points,
            'boundary_values': boundary_values,
            'initial_points': initial_points,
            'initial_values': initial_values,
            'data_points': data_points,
            'data_values': data_values
        }
        
        return data_dict
    
    def create_loss_function(self, data_dict: Dict[str, torch.Tensor]):
        """创建损失函数闭包"""
        def loss_fn():
            # 生成物理参数
            params = self.data_generator.generate_physics_parameters(data_dict['collocation_points'])
            
            # 检查参数中的NaN值
            for name, param in params.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Warning: NaN or Inf detected in physics parameter {name}")
                    params[name] = torch.where(torch.isnan(param) | torch.isinf(param),
                                             torch.tensor(1e-6, device=param.device), param)
            
            # 计算损失
            total_loss, loss_components = self.loss_fn(self.model, data_dict, params)
            
            # 检查损失中的NaN值
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN or Inf detected in total loss: {total_loss.item()}")
                total_loss = torch.tensor(1e6, device=total_loss.device, requires_grad=True)
            
            for name, component in loss_components.items():
                if torch.isnan(component) or torch.isinf(component):
                    print(f"Warning: NaN or Inf detected in loss component {name}: {component.item()}")
                    loss_components[name] = torch.tensor(0.0, device=component.device)
            
            return total_loss, loss_components
        
        return loss_fn
    
    def train_epoch(self, data_dict: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 创建损失函数
        loss_fn = self.create_loss_function(data_dict)
        
        # 计算损失和梯度
        total_loss, loss_components = loss_fn()
        
        # 自适应权重调整
        if self.config.get('use_adaptive_weighting', True) and epoch % 10 == 0:
            current_weights = self.loss_fn.get_weights()
            new_weights = self.adaptive_weighting.update_weights(
                loss_components, list(self.model.parameters()), current_weights
            )
            self.loss_fn.update_weights(new_weights)
            self.training_history['weights_history'].append(new_weights.copy())
        
        # 记录损失
        loss_dict = {
            'total': total_loss.item(),
            'residual': loss_components['residual'].item(),
            'boundary': loss_components['boundary'].item(),
            'initial': loss_components['initial'].item(),
            'data': loss_components['data'].item()
        }
        
        # 检查损失中的NaN值
        for name, value in loss_dict.items():
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: NaN or Inf detected in {name} loss: {value}")
                loss_dict[name] = 1e6 if name == 'total' else 0.0
        
        return loss_dict
    
    def train(self, max_epochs: int = 10000, verbose: bool = True) -> Dict[str, Any]:
        """训练PINN模型
        
        Args:
            max_epochs: 最大训练轮数
            verbose: 是否打印训练信息
        
        Returns:
            training_results: 训练结果
        """
        if verbose:
            print("Preparing training data...")
        
        # 准备数据
        data_dict = self.prepare_data()
        
        if verbose:
            print(f"Data prepared:")
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        
        # 创建损失函数
        loss_fn = self.create_loss_function(data_dict)
        
        if verbose:
            print("Starting training...")
        
        # 使用两阶段优化
        optimization_history = self.optimizer.optimize(loss_fn, verbose=verbose)
        
        # 更新训练历史
        self.training_history['total_losses'].extend(optimization_history['total_losses'])
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            final_loss, final_components = loss_fn()
            
        training_results = {
            'final_loss': final_loss.item(),
            'final_components': {k: v.item() for k, v in final_components.items()},
            'optimization_history': optimization_history,
            'training_history': self.training_history
        }
        
        if verbose:
            print(f"Training completed!")
            print(f"Final loss: {final_loss.item():.6e}")
            for name, component_loss in final_components.items():
                print(f"  {name}: {component_loss.item():.6e}")
        
        return training_results
    
    def evaluate(self, test_points: torch.Tensor, test_values: torch.Tensor) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            test_points: 测试点
            test_values: 测试值
        
        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        
        with torch.no_grad():
            # 归一化测试数据
            if self.config.get('normalize_inputs', True):
                test_points = self.input_normalizer.transform(test_points)
            
            # 模型预测
            predictions = self.model(test_points)
            
            # 反归一化预测结果
            if self.config.get('normalize_outputs', True):
                predictions = self.output_normalizer.inverse_transform(predictions)
            
            # 检查预测结果中的NaN值
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print(f"Warning: NaN or Inf detected in predictions")
                predictions = torch.where(torch.isnan(predictions) | torch.isinf(predictions),
                                        torch.tensor(0.0, device=predictions.device), predictions)
            
            # 计算指标
            mse = torch.mean((predictions - test_values) ** 2).item()
            mae = torch.mean(torch.abs(predictions - test_values)).item()
            rmse = np.sqrt(mse)
            
            # 相对误差
            relative_error = torch.mean(torch.abs((predictions - test_values) / (test_values + 1e-8))).item()
            
            # R²分数
            ss_res = torch.sum((test_values - predictions) ** 2)
            ss_tot = torch.sum((test_values - torch.mean(test_values)) ** 2)
            r2_score = (1 - ss_res / ss_tot).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'relative_error': relative_error,
            'r2_score': r2_score
        }
        
        return metrics
    
    def predict(self, points: torch.Tensor) -> torch.Tensor:
        """模型预测
        
        Args:
            points: 输入点
        
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            # 检查输入中的NaN值
            if torch.isnan(points).any() or torch.isinf(points).any():
                print(f"Warning: NaN or Inf detected in input points")
                points = torch.where(torch.isnan(points) | torch.isinf(points),
                                   torch.tensor(0.0, device=points.device), points)
            
            # 归一化输入
            if self.config.get('normalize_inputs', True):
                points = self.input_normalizer.transform(points)
            
            # 预测
            predictions = self.model(points)
            
            # 反归一化输出
            if self.config.get('normalize_outputs', True):
                predictions = self.output_normalizer.inverse_transform(predictions)
            
            # 检查预测结果中的NaN值
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print(f"Warning: NaN or Inf detected in predictions")
                predictions = torch.where(torch.isnan(predictions) | torch.isinf(predictions),
                                        torch.tensor(0.0, device=predictions.device), predictions)
        
        return predictions
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'input_normalizer_stats': self.input_normalizer.stats if self.input_normalizer.fitted else None,
            'output_normalizer_stats': self.output_normalizer.stats if self.output_normalizer.fitted else None,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        # 恢复归一化器状态
        if checkpoint.get('input_normalizer_stats') is not None:
            self.input_normalizer.stats = checkpoint['input_normalizer_stats']
            self.input_normalizer.fitted = True
        
        if checkpoint.get('output_normalizer_stats') is not None:
            self.output_normalizer.stats = checkpoint['output_normalizer_stats']
            self.output_normalizer.fitted = True
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        if not self.training_history['total_losses']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].semilogy(self.training_history['total_losses'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 损失分量
        for name, losses in self.training_history['loss_components'].items():
            if losses:
                axes[0, 1].semilogy(losses, label=name)
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 权重历史
        if self.training_history['weights_history']:
            weights_array = np.array([list(w.values()) for w in self.training_history['weights_history']])
            weight_names = list(self.training_history['weights_history'][0].keys())
            
            for i, name in enumerate(weight_names):
                axes[1, 0].plot(weights_array[:, i], label=name)
            axes[1, 0].set_title('Loss Weights')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 学习率历史
        if self.training_history['learning_rates']:
            axes[1, 1].semilogy(self.training_history['learning_rates'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_solution(self, t_slice: float = 0.5, save_path: Optional[str] = None):
        """绘制解的可视化
        
        Args:
            t_slice: 时间切片
            save_path: 保存路径
        """
        # 创建网格点
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # 创建输入点
        points = []
        for i in range(len(x)):
            for j in range(len(y)):
                # [t, x, y, z, ne, Te, ni, Ti, Sflux, alpha_sun, alpha_ram, material_id]
                point = [t_slice, X[i, j], Y[i, j], 0.0,  # t, x, y, z
                        1e7, 3000, 1e7, 3000,  # ne, Te, ni, Ti
                        1361, 0.0, 0.0, 0]  # Sflux, alpha_sun, alpha_ram, material_id
                points.append(point)
        
        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # 预测
        predictions = self.predict(points)
        predictions = predictions.cpu().numpy().reshape(len(x), len(y))
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X, Y, predictions, levels=50, cmap='viridis')
        ax.contour(X, Y, predictions, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Surface Potential (V)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Surface Potential at t = {t_slice}')
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()