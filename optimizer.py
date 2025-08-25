import torch
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from scipy.optimize import minimize

class TwoStageOptimizer:
    """两阶段优化器：Adam + L-BFGS"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        Args:
            model: PINN模型
            config: 优化器配置
        """
        self.model = model
        self.config = config
        
        # Adam优化器配置
        self.adam_config = config.get('adam', {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
            'max_iter': 5000
        })
        
        # L-BFGS优化器配置
        self.lbfgs_config = config.get('lbfgs', {
            'lr': 1.0,
            'max_iter': 1000,
            'max_eval': None,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'history_size': 100,
            'line_search_fn': 'strong_wolfe'
        })
        
        # 创建优化器
        self.adam_optimizer = None
        self.lbfgs_optimizer = None
        self._create_optimizers()
        
        # 训练历史
        self.training_history = {
            'adam_losses': [],
            'lbfgs_losses': [],
            'adam_iterations': 0,
            'lbfgs_iterations': 0
        }
        
    def _create_optimizers(self):
        """创建优化器"""
        # Adam优化器
        self.adam_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.adam_config['lr'],
            betas=self.adam_config['betas'],
            eps=self.adam_config['eps'],
            weight_decay=self.adam_config['weight_decay']
        )
        
        # L-BFGS优化器
        self.lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_config['lr'],
            max_iter=self.lbfgs_config['max_iter'],
            max_eval=None,#,self.lbfgs_config['max_eval'],
            tolerance_grad=self.lbfgs_config['tolerance_grad'],
            tolerance_change=self.lbfgs_config['tolerance_change'],
            history_size=self.lbfgs_config['history_size'],
            line_search_fn=self.lbfgs_config['line_search_fn']
        )
    
    def adam_step(self, loss_fn: Callable, verbose: bool = False) -> float:
        """执行一步Adam优化
        
        Args:
            loss_fn: 损失函数，返回(loss, loss_components)
            verbose: 是否打印详细信息
        
        Returns:
            loss_value: 损失值
        """
        self.adam_optimizer.zero_grad()
        
        # 计算损失
        loss, loss_components = loss_fn()
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选）
        if self.config.get('gradient_clipping', False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('max_grad_norm', 1.0)
            )
        
        # 优化步骤
        self.adam_optimizer.step()
        
        loss_value = loss.item()
        self.training_history['adam_losses'].append(loss_value)
        self.training_history['adam_iterations'] += 1
        
        if verbose:
            print(f"Adam Iter {self.training_history['adam_iterations']}: Loss = {loss_value:.6e}")
            for name, component_loss in loss_components.items():
                print(f"  {name}: {component_loss.item():.6e}")
        
        return loss_value
    
    def adam_optimize(self, loss_fn: Callable, max_iter: Optional[int] = None, 
                     tolerance: float = 1e-6, verbose: bool = True) -> List[float]:
        """Adam优化阶段
        
        Args:
            loss_fn: 损失函数
            max_iter: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否打印信息
        
        Returns:
            losses: 损失历史
        """
        if max_iter is None:
            max_iter = self.adam_config['max_iter']
        
        losses = []
        prev_loss = float('inf')
        
        if verbose:
            print("Starting Adam optimization...")
        
        for i in range(max_iter):
            loss_value = self.adam_step(loss_fn, verbose and (i % 100 == 0))
            losses.append(loss_value)
            
            # 检查收敛
            if abs(prev_loss - loss_value) < tolerance:
                if verbose:
                    print(f"Adam converged at iteration {i+1}")
                break
            
            prev_loss = loss_value
        
        if verbose:
            print(f"Adam optimization completed. Final loss: {losses[-1]:.6e}")
        
        return losses
    
    def lbfgs_optimize(self, loss_fn: Callable, verbose: bool = True) -> List[float]:
        """L-BFGS优化阶段
        
        Args:
            loss_fn: 损失函数
            verbose: 是否打印信息
        
        Returns:
            losses: 损失历史
        """
        losses = []
        
        def closure():
            self.lbfgs_optimizer.zero_grad()
            loss, loss_components = loss_fn()
            loss.backward()
            
            loss_value = loss.item()
            losses.append(loss_value)
            self.training_history['lbfgs_losses'].append(loss_value)
            self.training_history['lbfgs_iterations'] += 1
            
            if verbose and (len(losses) % 10 == 0):
                print(f"L-BFGS Iter {len(losses)}: Loss = {loss_value:.6e}")
                for name, component_loss in loss_components.items():
                    print(f"  {name}: {component_loss.item():.6e}")
            
            return loss
        
        if verbose:
            print("Starting L-BFGS optimization...")
        
        # 执行L-BFGS优化
        self.lbfgs_optimizer.step(closure)
        
        if verbose:
            print(f"L-BFGS optimization completed. Final loss: {losses[-1]:.6e}")
        
        return losses
    
    def optimize(self, loss_fn: Callable, verbose: bool = True) -> Dict[str, List[float]]:
        """执行两阶段优化
        
        Args:
            loss_fn: 损失函数
            verbose: 是否打印信息
        
        Returns:
            optimization_history: 优化历史
        """
        # 第一阶段：Adam优化
        adam_losses = self.adam_optimize(loss_fn, verbose=verbose)
        
        # 第二阶段：L-BFGS优化
        lbfgs_losses = self.lbfgs_optimize(loss_fn, verbose=verbose)
        
        optimization_history = {
            'adam_losses': adam_losses,
            'lbfgs_losses': lbfgs_losses,
            'total_losses': adam_losses + lbfgs_losses
        }
        
        return optimization_history
    
    def get_training_history(self) -> Dict[str, Any]:
        """获取训练历史"""
        return self.training_history.copy()
    
    def reset_history(self):
        """重置训练历史"""
        self.training_history = {
            'adam_losses': [],
            'lbfgs_losses': [],
            'adam_iterations': 0,
            'lbfgs_iterations': 0
        }

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, scheduler_type: str = 'exponential', **kwargs):
        """
        Args:
            optimizer: PyTorch优化器
            scheduler_type: 调度器类型
            **kwargs: 调度器参数
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'step':
            step_size = kwargs.get('step_size', 1000)
            gamma = kwargs.get('gamma', 0.5)
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 5000)
            eta_min = kwargs.get('eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'plateau':
            factor = kwargs.get('factor', 0.5)
            patience = kwargs.get('patience', 100)
            threshold = kwargs.get('threshold', 1e-4)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, threshold=threshold
            )
        else:
            self.scheduler = None
    
    def step(self, loss: Optional[float] = None):
        """执行调度步骤"""
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau':
                if loss is not None:
                    self.scheduler.step(loss)
            else:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 100, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        """
        Args:
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, loss: float, model: torch.nn.Module) -> bool:
        """
        Args:
            loss: 当前损失
            model: 模型
        
        Returns:
            early_stop: 是否应该早停
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop