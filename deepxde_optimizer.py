import numpy as np
import torch
import torch.optim as optim
import deepxde as dde
from typing import Dict, List, Tuple, Optional, Callable, Any
import time
import matplotlib.pyplot as plt

class TwoStageOptimizer:
    """两阶段优化器：Adam + L-BFGS"""
    
    def __init__(self, 
                 model: dde.Model,
                 adam_config: Dict[str, Any] = None,
                 lbfgs_config: Dict[str, Any] = None):
        """
        Args:
            model: DeepXDE模型
            adam_config: Adam优化器配置
            lbfgs_config: L-BFGS优化器配置
        """
        self.model = model
        
        # 默认Adam配置
        self.adam_config = adam_config or {
            'learning_rate': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'iterations': 10000,
            'display_every': 1000
        }
        
        # 默认L-BFGS配置
        self.lbfgs_config = lbfgs_config or {
            'learning_rate': 1.0,
            'max_iter': 5000,
            'max_eval': None,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'history_size': 100,
            'display_every': 500
        }
        
        # 训练历史
        self.training_history = {
            'adam': {'loss': [], 'iterations': [], 'time': []},
            'lbfgs': {'loss': [], 'iterations': [], 'time': []}
        }
        
        # 当前阶段
        self.current_stage = None
        self.stage_start_time = None
    
    def train_adam_stage(self, 
                        loss_weights: Optional[Dict[str, float]] = None,
                        callbacks: Optional[List[Callable]] = None) -> Tuple[Any, Any]:
        """Adam优化阶段
        
        Args:
            loss_weights: 损失权重
            callbacks: 回调函数列表
        
        Returns:
            损失历史和训练状态
        """
        print("\n" + "="*50)
        print("Starting Adam Optimization Stage")
        print("="*50)
        
        self.current_stage = 'adam'
        self.stage_start_time = time.time()
        
        # 编译模型 - Adam阶段
        self.model.compile(
            optimizer='adam',
            lr=self.adam_config['learning_rate'],
            loss_weights=loss_weights
        )
        
        # 训练
        losshistory, train_state = self.model.train(
            iterations=self.adam_config['iterations'],
            display_every=self.adam_config['display_every'],
            callbacks=callbacks
        )
        
        # 记录训练历史
        self._record_training_history('adam', losshistory)
        
        stage_time = time.time() - self.stage_start_time
        print(f"\nAdam stage completed in {stage_time:.2f} seconds")
        print(f"Final Adam loss: {losshistory.loss_train[-1]:.6e}")
        
        return losshistory, train_state
    
    def train_lbfgs_stage(self, 
                         loss_weights: Optional[Dict[str, float]] = None,
                         callbacks: Optional[List[Callable]] = None) -> Tuple[Any, Any]:
        """L-BFGS优化阶段
        
        Args:
            loss_weights: 损失权重
            callbacks: 回调函数列表
        
        Returns:
            损失历史和训练状态
        """
        print("\n" + "="*50)
        print("Starting L-BFGS Optimization Stage")
        print("="*50)
        
        self.current_stage = 'lbfgs'
        self.stage_start_time = time.time()
        
        # 编译模型 - L-BFGS阶段
        self.model.compile(
            optimizer='L-BFGS',
            lr=self.lbfgs_config['learning_rate'],
            loss_weights=loss_weights
        )
        
        # 训练
        losshistory, train_state = self.model.train(
            iterations=self.lbfgs_config['max_iter'],
            display_every=self.lbfgs_config['display_every'],
            callbacks=callbacks
        )
        
        # 记录训练历史
        self._record_training_history('lbfgs', losshistory)
        
        stage_time = time.time() - self.stage_start_time
        print(f"\nL-BFGS stage completed in {stage_time:.2f} seconds")
        print(f"Final L-BFGS loss: {losshistory.loss_train[-1]:.6e}")
        
        return losshistory, train_state
    
    def train_full_pipeline(self, 
                           loss_weights: Optional[Dict[str, float]] = None,
                           callbacks: Optional[List[Callable]] = None,
                           save_path: Optional[str] = None) -> Dict[str, Tuple[Any, Any]]:
        """完整的两阶段训练流程
        
        Args:
            loss_weights: 损失权重
            callbacks: 回调函数列表
            save_path: 模型保存路径
        
        Returns:
            两个阶段的训练结果
        """
        total_start_time = time.time()
        
        print("\n" + "="*60)
        print("PINN Two-Stage Training Pipeline")
        print("="*60)
        
        results = {}
        
        # 阶段1: Adam优化
        try:
            adam_history, adam_state = self.train_adam_stage(
                loss_weights=loss_weights,
                callbacks=callbacks
            )
            results['adam'] = (adam_history, adam_state)
        except Exception as e:
            print(f"Error in Adam stage: {e}")
            return results
        
        # 阶段2: L-BFGS优化
        try:
            lbfgs_history, lbfgs_state = self.train_lbfgs_stage(
                loss_weights=loss_weights,
                callbacks=callbacks
            )
            results['lbfgs'] = (lbfgs_history, lbfgs_state)
        except Exception as e:
            print(f"Error in L-BFGS stage: {e}")
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*60)
        print("Training Pipeline Summary")
        print("="*60)
        print(f"Total training time: {total_time:.2f} seconds")
        
        if 'adam' in results:
            print(f"Adam iterations: {len(results['adam'][0].loss_train)}")
            print(f"Final Adam loss: {results['adam'][0].loss_train[-1]:.6e}")
        
        if 'lbfgs' in results:
            print(f"L-BFGS iterations: {len(results['lbfgs'][0].loss_train)}")
            print(f"Final L-BFGS loss: {results['lbfgs'][0].loss_train[-1]:.6e}")
        
        # 保存模型
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to: {save_path}")
        
        return results
    
    def _record_training_history(self, stage: str, losshistory: Any):
        """记录训练历史"""
        self.training_history[stage]['loss'] = losshistory.loss_train.copy()
        self.training_history[stage]['iterations'] = list(range(len(losshistory.loss_train)))
        
        # 记录时间信息
        if hasattr(losshistory, 'time'):
            self.training_history[stage]['time'] = losshistory.time.copy()
        else:
            # 如果没有时间信息，使用估计值
            stage_time = time.time() - self.stage_start_time
            num_iterations = len(losshistory.loss_train)
            self.training_history[stage]['time'] = [
                i * stage_time / num_iterations for i in range(num_iterations)
            ]
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制损失曲线
        adam_offset = 0
        lbfgs_offset = 0
        
        if self.training_history['adam']['loss']:
            adam_iterations = np.array(self.training_history['adam']['iterations'])
            adam_loss = np.array(self.training_history['adam']['loss'])
            ax1.plot(adam_iterations, adam_loss, 'b-', label='Adam', linewidth=2)
            adam_offset = len(adam_iterations)
        
        if self.training_history['lbfgs']['loss']:
            lbfgs_iterations = np.array(self.training_history['lbfgs']['iterations']) + adam_offset
            lbfgs_loss = np.array(self.training_history['lbfgs']['loss'])
            ax1.plot(lbfgs_iterations, lbfgs_loss, 'r-', label='L-BFGS', linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Two-Stage Training Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 绘制时间-损失曲线
        adam_time_offset = 0
        
        if self.training_history['adam']['time'] and self.training_history['adam']['loss']:
            adam_time = np.array(self.training_history['adam']['time'])
            adam_loss = np.array(self.training_history['adam']['loss'])
            ax2.plot(adam_time, adam_loss, 'b-', label='Adam', linewidth=2)
            adam_time_offset = adam_time[-1] if len(adam_time) > 0 else 0
        
        if self.training_history['lbfgs']['time'] and self.training_history['lbfgs']['loss']:
            lbfgs_time = np.array(self.training_history['lbfgs']['time']) + adam_time_offset
            lbfgs_loss = np.array(self.training_history['lbfgs']['loss'])
            ax2.plot(lbfgs_time, lbfgs_loss, 'r-', label='L-BFGS', linewidth=2)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'adam': {},
            'lbfgs': {},
            'total': {}
        }
        
        # Adam阶段摘要
        if self.training_history['adam']['loss']:
            adam_loss = self.training_history['adam']['loss']
            summary['adam'] = {
                'iterations': len(adam_loss),
                'initial_loss': adam_loss[0],
                'final_loss': adam_loss[-1],
                'loss_reduction': adam_loss[0] - adam_loss[-1],
                'relative_reduction': (adam_loss[0] - adam_loss[-1]) / adam_loss[0]
            }
            
            if self.training_history['adam']['time']:
                summary['adam']['training_time'] = self.training_history['adam']['time'][-1]
        
        # L-BFGS阶段摘要
        if self.training_history['lbfgs']['loss']:
            lbfgs_loss = self.training_history['lbfgs']['loss']
            summary['lbfgs'] = {
                'iterations': len(lbfgs_loss),
                'initial_loss': lbfgs_loss[0],
                'final_loss': lbfgs_loss[-1],
                'loss_reduction': lbfgs_loss[0] - lbfgs_loss[-1],
                'relative_reduction': (lbfgs_loss[0] - lbfgs_loss[-1]) / lbfgs_loss[0]
            }
            
            if self.training_history['lbfgs']['time']:
                summary['lbfgs']['training_time'] = self.training_history['lbfgs']['time'][-1]
        
        # 总体摘要
        total_iterations = 0
        total_time = 0
        initial_loss = None
        final_loss = None
        
        if summary['adam']:
            total_iterations += summary['adam']['iterations']
            total_time += summary['adam'].get('training_time', 0)
            initial_loss = summary['adam']['initial_loss']
        
        if summary['lbfgs']:
            total_iterations += summary['lbfgs']['iterations']
            total_time += summary['lbfgs'].get('training_time', 0)
            final_loss = summary['lbfgs']['final_loss']
            if initial_loss is None:
                initial_loss = summary['lbfgs']['initial_loss']
        
        if initial_loss is not None and final_loss is not None:
            summary['total'] = {
                'total_iterations': total_iterations,
                'total_time': total_time,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'total_loss_reduction': initial_loss - final_loss,
                'total_relative_reduction': (initial_loss - final_loss) / initial_loss
            }
        
        return summary
    
    def save_training_history(self, file_path: str):
        """保存训练历史"""
        np.savez(file_path, 
                adam_loss=self.training_history['adam']['loss'],
                adam_iterations=self.training_history['adam']['iterations'],
                adam_time=self.training_history['adam']['time'],
                lbfgs_loss=self.training_history['lbfgs']['loss'],
                lbfgs_iterations=self.training_history['lbfgs']['iterations'],
                lbfgs_time=self.training_history['lbfgs']['time'])
        print(f"Training history saved to {file_path}")
    
    def load_training_history(self, file_path: str):
        """加载训练历史"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            self.training_history['adam']['loss'] = data['adam_loss'].tolist()
            self.training_history['adam']['iterations'] = data['adam_iterations'].tolist()
            self.training_history['adam']['time'] = data['adam_time'].tolist()
            
            self.training_history['lbfgs']['loss'] = data['lbfgs_loss'].tolist()
            self.training_history['lbfgs']['iterations'] = data['lbfgs_iterations'].tolist()
            self.training_history['lbfgs']['time'] = data['lbfgs_time'].tolist()
            
            print(f"Training history loaded from {file_path}")
        except Exception as e:
            print(f"Error loading training history: {e}")

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, 
                 scheduler_type: str = 'exponential',
                 scheduler_params: Dict[str, float] = None):
        """
        Args:
            scheduler_type: 调度器类型 ('exponential', 'step', 'cosine')
            scheduler_params: 调度器参数
        """
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # 默认参数
        if scheduler_type == 'exponential':
            self.scheduler_params.setdefault('decay_rate', 0.95)
            self.scheduler_params.setdefault('decay_steps', 1000)
        elif scheduler_type == 'step':
            self.scheduler_params.setdefault('step_size', 2000)
            self.scheduler_params.setdefault('gamma', 0.5)
        elif scheduler_type == 'cosine':
            self.scheduler_params.setdefault('T_max', 10000)
            self.scheduler_params.setdefault('eta_min', 1e-6)
    
    def get_learning_rate(self, iteration: int, base_lr: float) -> float:
        """获取当前迭代的学习率
        
        Args:
            iteration: 当前迭代次数
            base_lr: 基础学习率
        
        Returns:
            调整后的学习率
        """
        if self.scheduler_type == 'exponential':
            decay_rate = self.scheduler_params['decay_rate']
            decay_steps = self.scheduler_params['decay_steps']
            return base_lr * (decay_rate ** (iteration // decay_steps))
        
        elif self.scheduler_type == 'step':
            step_size = self.scheduler_params['step_size']
            gamma = self.scheduler_params['gamma']
            return base_lr * (gamma ** (iteration // step_size))
        
        elif self.scheduler_type == 'cosine':
            T_max = self.scheduler_params['T_max']
            eta_min = self.scheduler_params['eta_min']
            return eta_min + (base_lr - eta_min) * (
                1 + np.cos(np.pi * iteration / T_max)
            ) / 2
        
        else:
            return base_lr

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, 
                 patience: int = 1000,
                 min_delta: float = 1e-6,
                 restore_best_weights: bool = True):
        """
        Args:
            patience: 容忍的迭代次数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_iteration = 0
        self.best_weights = None
    
    def __call__(self, current_loss: float, model: dde.Model) -> bool:
        """检查是否应该早停
        
        Args:
            current_loss: 当前损失值
            model: 模型
        
        Returns:
            是否应该停止训练
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                # 保存当前最佳权重
                # 注意：这里需要根据DeepXDE的具体API来实现权重保存
                pass
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_iteration = self.wait
                if self.restore_best_weights and self.best_weights is not None:
                    # 恢复最佳权重
                    # 注意：这里需要根据DeepXDE的具体API来实现权重恢复
                    pass
                return True
        
        return False
    
    def reset(self):
        """重置早停状态"""
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_iteration = 0
        self.best_weights = None