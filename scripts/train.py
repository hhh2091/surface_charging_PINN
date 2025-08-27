# 训练脚本
# 实现两阶段优化策略的PINN训练

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.networks import create_pinn_model, print_model_summary
from pinn_model.dataset import create_data_loader
from pinn_model.physics import validate_physics_constants

class PINNTrainer:
    """
    PINN训练器
    
    实现两阶段优化策略：
    1. Adam优化器快速收敛
    2. L-BFGS优化器精确收敛
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.training_config = config['training']
        
        # 创建模型
        self.model = create_pinn_model(config).to(device)
        
        # 损失函数权重
        self.loss_weights = self.training_config['loss_weights']
        self.adaptive_weights = self.training_config['adaptive_weights']
        
        # 训练历史
        self.train_history = {
            'total_loss': [],
            'physics_loss': [],
            'data_loss': [],
            'initial_loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # 最佳模型状态
        self.best_loss = float('inf')
        self.best_model_state = None
        
        # 检查点目录
        self.checkpoint_dir = Path(self.training_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"PINN trainer initialized on device: {device}")
        print_model_summary(self.model)
    
    def compute_physics_loss(self, physics_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算物理残差损失
        
        Args:
            physics_batch: 物理约束批次数据
            
        Returns:
            physics_loss: 物理损失
        """
        inputs = physics_batch['inputs'].to(self.device)
        inputs.requires_grad_(True)
        
        # 计算物理残差
        residual = self.model.compute_physics_residual(inputs)
        
        # 计算MSE损失
        physics_loss = torch.mean(residual ** 2)
        
        return physics_loss
    
    def compute_data_loss(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算数据拟合损失
        
        Args:
            data_batch: 数据批次
            
        Returns:
            data_loss: 数据损失
        """
        inputs = data_batch['inputs'].to(self.device)
        targets = data_batch['outputs'].to(self.device)
        
        # 模型预测
        predictions = self.model.predict(inputs)
        
        # 计算MSE损失
        data_loss = torch.mean((predictions - targets) ** 2)
        
        return data_loss
    
    def compute_initial_condition_loss(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算初始条件损失
        
        Args:
            data_batch: 数据批次
            
        Returns:
            initial_loss: 初始条件损失
        """
        inputs = data_batch['inputs'].to(self.device)
        targets = data_batch['outputs'].to(self.device)
        
        # 找到t=0的点
        t_values = inputs[:, 0]
        initial_mask = torch.abs(t_values - t_values.min()) < 1e-6
        
        if initial_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 初始条件：V(t=0) = 0
        initial_inputs = inputs[initial_mask]
        initial_predictions = self.model.predict(initial_inputs)
        initial_targets = torch.zeros_like(initial_predictions)
        
        # 计算MSE损失
        initial_loss = torch.mean((initial_predictions - initial_targets) ** 2)
        
        return initial_loss
    
    def compute_total_loss(self, physics_batch: Dict[str, torch.Tensor],
                          data_batch: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            physics_batch: 物理约束批次
            data_batch: 数据批次（可选）
            
        Returns:
            total_loss: 总损失
            loss_components: 损失分量字典
        """
        # 物理损失
        physics_loss = self.compute_physics_loss(physics_batch)
        
        # 数据损失和初始条件损失
        if data_batch is not None and 'outputs' in data_batch:
            data_loss = self.compute_data_loss(data_batch)
            initial_loss = self.compute_initial_condition_loss(data_batch)
        else:
            data_loss = torch.tensor(0.0, device=self.device)
            initial_loss = torch.tensor(0.0, device=self.device)
        
        # 加权总损失
        total_loss = (
            self.loss_weights['lambda_residual'] * physics_loss +
            self.loss_weights['lambda_data'] * data_loss +
            self.loss_weights['lambda_initial'] * initial_loss
        )
        
        loss_components = {
            'physics_loss': physics_loss,
            'data_loss': data_loss,
            'initial_loss': initial_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components
    
    def update_adaptive_weights(self, loss_components: Dict[str, torch.Tensor]):
        """
        更新自适应权重（梯度归一化）
        
        Args:
            loss_components: 损失分量字典
        """
        if not self.adaptive_weights['enabled']:
            return
        
        # 计算各损失项的梯度范数
        physics_grad_norm = 0.0
        data_grad_norm = 0.0
        initial_grad_norm = 0.0
        
        # 物理损失梯度
        if loss_components['physics_loss'].requires_grad:
            physics_grads = torch.autograd.grad(
                loss_components['physics_loss'], self.model.parameters(),
                retain_graph=True, create_graph=False, allow_unused=True
            )
            physics_grad_norm = sum(g.norm().item() for g in physics_grads if g is not None)
        
        # 数据损失梯度
        if loss_components['data_loss'].requires_grad and loss_components['data_loss'].item() > 0:
            data_grads = torch.autograd.grad(
                loss_components['data_loss'], self.model.parameters(),
                retain_graph=True, create_graph=False, allow_unused=True
            )
            data_grad_norm = sum(g.norm().item() for g in data_grads if g is not None)
        
        # 初始条件损失梯度
        if loss_components['initial_loss'].requires_grad and loss_components['initial_loss'].item() > 0:
            initial_grads = torch.autograd.grad(
                loss_components['initial_loss'], self.model.parameters(),
                retain_graph=True, create_graph=False, allow_unused=True
            )
            initial_grad_norm = sum(g.norm().item() for g in initial_grads if g is not None)
        
        # 更新权重（梯度归一化）
        total_grad_norm = physics_grad_norm + data_grad_norm + initial_grad_norm
        if total_grad_norm > 0:
            self.loss_weights['lambda_residual'] = physics_grad_norm / total_grad_norm
            self.loss_weights['lambda_data'] = data_grad_norm / total_grad_norm if data_grad_norm > 0 else 0.1
            self.loss_weights['lambda_initial'] = initial_grad_norm / total_grad_norm if initial_grad_norm > 0 else 0.1
    
    def train_adam_phase(self, data_loader) -> None:
        """
        Adam优化阶段
        
        Args:
            data_loader: 数据加载器
        """
        logging.info("Starting Adam optimization phase...")
        
        # 创建Adam优化器
        adam_config = self.training_config['adam_phase']
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(adam_config['learning_rate']),
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000, verbose=True
        )
        
        # 训练循环
        self.model.train()
        num_iterations = int(adam_config['iterations'])
        
        # 创建进度条
        pbar = tqdm(range(num_iterations), desc="Adam Phase")
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # 获取批次数据
            # try:
            # 物理约束数据
            physics_batch = next(iter(data_loader.physics_loader))
            
            # 监督数据（如果有）
            data_batch = None
            if data_loader.train_loader is not None:
                try:
                    data_batch = next(iter(data_loader.train_loader))
                except:
                    pass
            
            # 计算损失
            total_loss, loss_components = self.compute_total_loss(physics_batch, data_batch)
            
            # 更新自适应权重（在backward之前）
            if iteration % self.adaptive_weights['update_frequency'] == 0:
                self.update_adaptive_weights(loss_components)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            
            # 更新学习率
            scheduler.step(total_loss)
            
            # 记录训练历史
            current_lr = optimizer.param_groups[0]['lr']
            self.train_history['total_loss'].append(total_loss.item())
            self.train_history['physics_loss'].append(loss_components['physics_loss'].item())
            self.train_history['data_loss'].append(loss_components['data_loss'].item())
            self.train_history['initial_loss'].append(loss_components['initial_loss'].item())
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['epoch'].append(iteration)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'Physics': f'{loss_components["physics_loss"].item():.6f}',
                'Data': f'{loss_components["data_loss"].item():.6f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # 保存最佳模型
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_model_state = self.model.state_dict().copy()
            
            # 定期保存检查点
            if iteration % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(iteration, 'adam')
            
            # except Exception as e:
            #     logging.error(f"Error in Adam iteration {iteration}: {e}")
            #     continue
        
        logging.info(f"Adam phase completed. Best loss: {self.best_loss:.6f}")
    
    def train_lbfgs_phase(self, data_loader) -> None:
        """
        L-BFGS优化阶段
        
        Args:
            data_loader: 数据加载器
        """
        logging.info("Starting L-BFGS optimization phase...")
        
        # 加载最佳Adam模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 创建L-BFGS优化器
        lbfgs_config = self.training_config['lbfgs_phase']
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lbfgs_config['learning_rate'],
            max_iter=lbfgs_config['max_iter'],
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100
        )
        
        # 预先获取数据批次
        physics_batch = next(iter(data_loader.physics_loader))
        data_batch = None
        if data_loader.train_loader is not None:
            try:
                data_batch = next(iter(data_loader.train_loader))
            except:
                pass
        
        # 训练循环
        self.model.train()
        num_iterations = lbfgs_config['iterations']
        
        # 创建进度条
        pbar = tqdm(range(num_iterations), desc="L-BFGS Phase")
        
        for iteration in pbar:
            def closure():
                optimizer.zero_grad()
                total_loss, loss_components = self.compute_total_loss(physics_batch, data_batch)
                
                # 更新自适应权重
                if iteration % self.adaptive_weights['update_frequency'] == 0:
                    self.update_adaptive_weights(loss_components)
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                optimizer.step()
                
                # 更新学习率
                scheduler.step(total_loss)
                
                # 记录训练历史
                current_lr = optimizer.param_groups[0]['lr']
                self.train_history['total_loss'].append(total_loss.item())
                self.train_history['physics_loss'].append(loss_components['physics_loss'].item())
                self.train_history['data_loss'].append(loss_components['data_loss'].item())
                self.train_history['initial_loss'].append(loss_components['initial_loss'].item())
                self.train_history['learning_rate'].append(lbfgs_config['learning_rate'])
                self.train_history['epoch'].append(len(self.train_history['epoch']) + iteration)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'Physics': f'{loss_components["physics_loss"].item():.6f}',
                    'Data': f'{loss_components["data_loss"].item():.6f}'
                })
                
                # 保存最佳模型
                if total_loss.item() < self.best_loss:
                    self.best_loss = total_loss.item()
                    self.best_model_state = self.model.state_dict().copy()
                
                # 定期保存检查点
                if iteration % self.training_config['save_frequency'] == 0:
                    self.save_checkpoint(iteration, 'lbfgs')
            
            # except Exception as e:
            #     logging.error(f"Error in L-BFGS iteration {iteration}: {e}")
            #     continue
        
        logging.info(f"L-BFGS phase completed. Final best loss: {self.best_loss:.6f}")
    
    def save_checkpoint(self, iteration: int, phase: str) -> None:
        """
        保存训练检查点
        
        Args:
            iteration: 当前迭代次数
            phase: 训练阶段
        """
        checkpoint = {
            'iteration': iteration,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model_state,
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{phase}_{iteration}.pth'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self) -> None:
        """
        保存最终训练好的模型
        """
        # 加载最佳模型状态
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 保存完整模型
        model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'best_loss': self.best_loss
        }, model_path)
        
        logging.info(f"Final model saved: {model_path}")
    
    def plot_training_history(self) -> None:
        """
        绘制训练历史
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True)
        
        # 损失分量
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['physics_loss'], label='Physics')
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['data_loss'], label='Data')
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['initial_loss'], label='Initial')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 最近1000次迭代的损失
        recent_epochs = self.train_history['epoch'][-1000:]
        recent_losses = self.train_history['total_loss'][-1000:]
        axes[1, 1].plot(recent_epochs, recent_losses)
        axes[1, 1].set_title('Recent Total Loss (Last 1000 iterations)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training history plot saved: {plot_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_level: str = 'INFO'):
    """
    设置日志
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description='Train PINN model for spacecraft charging')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--train_data', type=str, default='data/train_data.csv',
                       help='Training data path')
    parser.add_argument('--test_data', type=str, default='data/test_data.csv',
                       help='Test data path')
    parser.add_argument('--physics_samples', type=int, default=10000,
                       help='Number of physics constraint samples')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 加载配置
    config = load_config(args.config)
    
    # 验证物理常数
    if not validate_physics_constants(config):
        raise ValueError("Invalid physics constants in configuration")
    
    # 设置设备
    device_config = config.get('device', {})
    if device_config.get('use_gpu', True) and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}")
        logging.info(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    # 创建数据加载器
    data_loader = create_data_loader(config)
    
    # 加载数据集
    train_path = args.train_data if Path(args.train_data).exists() else None
    test_path = args.test_data if Path(args.test_data).exists() else None
    
    data_loader.load_datasets(
        train_path=train_path,
        test_path=test_path,
        physics_samples=args.physics_samples
    )
    
    # 创建数据加载器
    data_loader.create_data_loaders()
    
    # 设置模型归一化参数
    if data_loader.train_dataset is not None:
        input_stats, output_stats = data_loader.get_normalization_stats()
        # 这里可以设置模型的归一化参数，如果需要的话
    
    # 创建训练器
    trainer = PINNTrainer(config, device)
    
    # 恢复训练（如果指定）
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.train_history = checkpoint.get('train_history', trainer.train_history)
        trainer.best_loss = checkpoint.get('best_loss', float('inf'))
        trainer.best_model_state = checkpoint.get('best_model_state_dict')
        logging.info(f"Resumed training from {args.resume}")
    
    # 开始训练
    start_time = time.time()
    
    try:
        # Adam优化阶段
        trainer.train_adam_phase(data_loader)
        
        # L-BFGS优化阶段
        trainer.train_lbfgs_phase(data_loader)
        
        # 保存最终模型
        trainer.save_final_model()
        
        # 绘制训练历史
        trainer.plot_training_history()
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(len(trainer.train_history['epoch']), 'interrupted')
    
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    finally:
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        logging.info(f"Final best loss: {trainer.best_loss:.6f}")

if __name__ == '__main__':
    main()