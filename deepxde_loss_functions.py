import numpy as np
import torch
import torch.nn as nn
import deepxde as dde
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt

class AdaptiveLossWeighting:
    """自适应损失权重调整"""
    
    def __init__(self, 
                 method: str = 'gradient_normalization',
                 update_frequency: int = 100,
                 alpha: float = 0.9):
        """
        Args:
            method: 权重调整方法 ('gradient_normalization', 'loss_balancing', 'annealing')
            update_frequency: 权重更新频率
            alpha: 指数移动平均参数
        """
        self.method = method
        self.update_frequency = update_frequency
        self.alpha = alpha
        
        # 权重历史
        self.weight_history = []
        self.loss_history = []
        self.gradient_history = []
        
        # 当前权重
        self.current_weights = None
        
        # 梯度统计
        self.grad_norm_avg = None
        self.loss_avg = None
    
    def initialize_weights(self, loss_components: List[str]) -> Dict[str, float]:
        """初始化权重
        
        Args:
            loss_components: 损失组件名称列表
        
        Returns:
            初始权重字典
        """
        # 初始权重设置
        initial_weights = {
            'pde': 1.0,
            'bc': 1.0, 
            'ic': 1.0,
            'data': 1.0
        }
        
        # 只保留存在的损失组件
        self.current_weights = {k: v for k, v in initial_weights.items() 
                               if k in loss_components}
        
        return self.current_weights.copy()
    
    def update_weights(self, 
                      loss_values: Dict[str, float],
                      gradients: Dict[str, torch.Tensor],
                      iteration: int) -> Dict[str, float]:
        """更新权重
        
        Args:
            loss_values: 各损失组件的值
            gradients: 各损失组件的梯度
            iteration: 当前迭代次数
        
        Returns:
            更新后的权重
        """
        if iteration % self.update_frequency != 0:
            return self.current_weights.copy()
        
        if self.method == 'gradient_normalization':
            return self._gradient_normalization(loss_values, gradients)
        elif self.method == 'loss_balancing':
            return self._loss_balancing(loss_values)
        elif self.method == 'annealing':
            return self._annealing_schedule(iteration)
        else:
            return self.current_weights.copy()
    
    def _gradient_normalization(self, 
                               loss_values: Dict[str, float],
                               gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """梯度归一化方法"""
        # 计算梯度范数
        grad_norms = {}
        for key, grad in gradients.items():
            if grad is not None:
                grad_norms[key] = torch.norm(grad).item()
            else:
                grad_norms[key] = 0.0
        
        # 更新梯度范数的移动平均
        if self.grad_norm_avg is None:
            self.grad_norm_avg = grad_norms.copy()
        else:
            for key in grad_norms:
                self.grad_norm_avg[key] = (self.alpha * self.grad_norm_avg[key] + 
                                          (1 - self.alpha) * grad_norms[key])
        
        # 计算目标梯度范数 (所有组件的平均)
        target_norm = np.mean(list(self.grad_norm_avg.values()))
        
        # 更新权重
        new_weights = {}
        for key in self.current_weights:
            if self.grad_norm_avg[key] > 0:
                new_weights[key] = self.current_weights[key] * target_norm / self.grad_norm_avg[key]
            else:
                new_weights[key] = self.current_weights[key]
            
            # 限制权重范围
            new_weights[key] = np.clip(new_weights[key], 0.01, 100.0)
        
        self.current_weights = new_weights
        self.weight_history.append(new_weights.copy())
        
        return new_weights.copy()
    
    def _loss_balancing(self, loss_values: Dict[str, float]) -> Dict[str, float]:
        """损失平衡方法"""
        # 更新损失的移动平均
        if self.loss_avg is None:
            self.loss_avg = loss_values.copy()
        else:
            for key in loss_values:
                self.loss_avg[key] = (self.alpha * self.loss_avg[key] + 
                                     (1 - self.alpha) * loss_values[key])
        
        # 计算目标损失 (所有组件的几何平均)
        valid_losses = [v for v in self.loss_avg.values() if v > 0]
        if valid_losses:
            target_loss = np.exp(np.mean(np.log(valid_losses)))
        else:
            target_loss = 1.0
        
        # 更新权重
        new_weights = {}
        for key in self.current_weights:
            if self.loss_avg[key] > 0:
                new_weights[key] = self.current_weights[key] * target_loss / self.loss_avg[key]
            else:
                new_weights[key] = self.current_weights[key]
            
            # 限制权重范围
            new_weights[key] = np.clip(new_weights[key], 0.01, 100.0)
        
        self.current_weights = new_weights
        self.weight_history.append(new_weights.copy())
        
        return new_weights.copy()
    
    def _annealing_schedule(self, iteration: int) -> Dict[str, float]:
        """退火调度方法"""
        # 简单的退火调度：随着训练进行，逐渐增加数据损失的权重
        annealing_factor = min(1.0, iteration / 10000.0)
        
        new_weights = self.current_weights.copy()
        
        # 调整数据损失权重
        if 'data' in new_weights:
            new_weights['data'] = 1.0 + 9.0 * annealing_factor  # 从1增加到10
        
        # 调整PDE损失权重
        if 'pde' in new_weights:
            new_weights['pde'] = 10.0 * (1.0 - 0.5 * annealing_factor)  # 从10减少到5
        
        self.current_weights = new_weights
        self.weight_history.append(new_weights.copy())
        
        return new_weights.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy() if self.current_weights else {}
    
    def plot_weight_history(self, save_path: Optional[str] = None):
        """绘制权重历史"""
        if not self.weight_history:
            print("No weight history to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 提取权重数据
        iterations = np.arange(len(self.weight_history)) * self.update_frequency
        
        for key in self.weight_history[0].keys():
            weights = [w[key] for w in self.weight_history]
            plt.plot(iterations, weights, label=f'{key} weight', marker='o', markersize=3)
        
        plt.xlabel('Iteration')
        plt.ylabel('Weight Value')
        plt.title('Adaptive Loss Weight Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class PINNLossMonitor:
    """PINN损失监控器"""
    
    def __init__(self):
        self.loss_history = {
            'total': [],
            'pde': [],
            'bc': [],
            'ic': [],
            'data': []
        }
        self.weight_history = []
        self.iteration_history = []
    
    def update(self, 
              iteration: int,
              loss_values: Dict[str, float],
              weights: Dict[str, float]):
        """更新损失历史
        
        Args:
            iteration: 迭代次数
            loss_values: 损失值
            weights: 权重值
        """
        self.iteration_history.append(iteration)
        
        # 更新各组件损失
        for key in self.loss_history.keys():
            if key in loss_values:
                self.loss_history[key].append(loss_values[key])
            else:
                self.loss_history[key].append(0.0)
        
        self.weight_history.append(weights.copy())
    
    def plot_loss_history(self, save_path: Optional[str] = None):
        """绘制损失历史"""
        if not self.iteration_history:
            print("No loss history to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制损失曲线
        iterations = np.array(self.iteration_history)
        
        for key, values in self.loss_history.items():
            if any(v > 0 for v in values):  # 只绘制非零损失
                ax1.plot(iterations, values, label=f'{key} loss', linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('PINN Loss Components Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 绘制权重曲线
        if self.weight_history:
            weight_keys = list(self.weight_history[0].keys())
            for key in weight_keys:
                weights = [w.get(key, 0) for w in self.weight_history]
                ax2.plot(iterations, weights, label=f'{key} weight', 
                        marker='o', markersize=2, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Weight Value')
        ax2.set_title('Loss Weights Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """计算收敛指标"""
        if len(self.loss_history['total']) < 100:
            return {}
        
        metrics = {}
        
        # 计算最近100次迭代的损失变化率
        recent_losses = self.loss_history['total'][-100:]
        if len(recent_losses) > 1:
            relative_change = abs(recent_losses[-1] - recent_losses[0]) / abs(recent_losses[0])
            metrics['relative_change_100'] = relative_change
        
        # 计算损失的标准差 (稳定性指标)
        if len(recent_losses) > 10:
            metrics['stability_std'] = np.std(recent_losses[-50:])
        
        # 计算最终损失值
        metrics['final_total_loss'] = self.loss_history['total'][-1]
        metrics['final_pde_loss'] = self.loss_history['pde'][-1]
        
        return metrics
    
    def save_history(self, file_path: str):
        """保存损失历史"""
        history_data = {
            'iterations': self.iteration_history,
            'loss_history': self.loss_history,
            'weight_history': self.weight_history
        }
        
        np.savez(file_path, **history_data)
        print(f"Loss history saved to {file_path}")
    
    def load_history(self, file_path: str):
        """加载损失历史"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            self.iteration_history = data['iterations'].tolist()
            self.loss_history = data['loss_history'].item()
            self.weight_history = data['weight_history'].tolist()
            
            print(f"Loss history loaded from {file_path}")
        except Exception as e:
            print(f"Error loading loss history: {e}")

class CustomLossFunction:
    """自定义损失函数"""
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 reduction: str = 'mean'):
        """
        Args:
            loss_type: 损失类型 ('mse', 'mae', 'huber', 'focal')
            reduction: 归约方式 ('mean', 'sum', 'none')
        """
        self.loss_type = loss_type
        self.reduction = reduction
    
    def __call__(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            损失值
        """
        if self.loss_type == 'mse':
            loss = torch.nn.functional.mse_loss(
                predictions, targets, reduction=self.reduction
            )
        elif self.loss_type == 'mae':
            loss = torch.nn.functional.l1_loss(
                predictions, targets, reduction=self.reduction
            )
        elif self.loss_type == 'huber':
            loss = torch.nn.functional.huber_loss(
                predictions, targets, reduction=self.reduction, delta=1.0
            )
        elif self.loss_type == 'focal':
            loss = self._focal_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _focal_loss(self, predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   alpha: float = 1.0,
                   gamma: float = 2.0) -> torch.Tensor:
        """Focal损失 (用于处理困难样本)"""
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
        
        # 计算权重
        pt = torch.exp(-mse)
        focal_weight = alpha * (1 - pt) ** gamma
        
        focal_loss = focal_weight * mse
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LossScheduler:
    """损失调度器"""
    
    def __init__(self, 
                 schedule_type: str = 'exponential',
                 schedule_params: Dict[str, float] = None):
        """
        Args:
            schedule_type: 调度类型 ('exponential', 'linear', 'cosine')
            schedule_params: 调度参数
        """
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params or {}
        
        # 默认参数
        if schedule_type == 'exponential':
            self.schedule_params.setdefault('decay_rate', 0.95)
            self.schedule_params.setdefault('decay_steps', 1000)
        elif schedule_type == 'linear':
            self.schedule_params.setdefault('start_factor', 1.0)
            self.schedule_params.setdefault('end_factor', 0.1)
            self.schedule_params.setdefault('total_steps', 10000)
    
    def get_loss_weights(self, iteration: int) -> Dict[str, float]:
        """获取当前迭代的损失权重
        
        Args:
            iteration: 当前迭代次数
        
        Returns:
            损失权重字典
        """
        if self.schedule_type == 'exponential':
            return self._exponential_schedule(iteration)
        elif self.schedule_type == 'linear':
            return self._linear_schedule(iteration)
        elif self.schedule_type == 'cosine':
            return self._cosine_schedule(iteration)
        else:
            return {'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'data': 1.0}
    
    def _exponential_schedule(self, iteration: int) -> Dict[str, float]:
        """指数衰减调度"""
        decay_rate = self.schedule_params['decay_rate']
        decay_steps = self.schedule_params['decay_steps']
        
        factor = decay_rate ** (iteration // decay_steps)
        
        return {
            'pde': 1.0,
            'bc': 1.0,
            'ic': 1.0,
            'data': factor  # 数据损失权重随时间衰减
        }
    
    def _linear_schedule(self, iteration: int) -> Dict[str, float]:
        """线性调度"""
        start_factor = self.schedule_params['start_factor']
        end_factor = self.schedule_params['end_factor']
        total_steps = self.schedule_params['total_steps']
        
        progress = min(iteration / total_steps, 1.0)
        factor = start_factor + (end_factor - start_factor) * progress
        
        return {
            'pde': 1.0,
            'bc': 1.0,
            'ic': 1.0,
            'data': factor
        }
    
    def _cosine_schedule(self, iteration: int) -> Dict[str, float]:
        """余弦调度"""
        total_steps = self.schedule_params.get('total_steps', 10000)
        
        progress = min(iteration / total_steps, 1.0)
        factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        return {
            'pde': 1.0,
            'bc': 1.0,
            'ic': 1.0,
            'data': factor
        }