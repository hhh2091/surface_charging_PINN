import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class PINNLoss(nn.Module):
    """PINN损失函数类"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super(PINNLoss, self).__init__()
        
        # 默认权重
        default_weights = {
            'residual': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'data': 1.0
        }
        
        self.weights = weights if weights is not None else default_weights
        
        # MSE损失函数
        self.mse_loss = nn.MSELoss()
        
        # 用于存储各个损失分量
        self.loss_components = {}
        
    def residual_loss(self, model, collocation_points: torch.Tensor, 
                     params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """物理残差损失 L_residual
        
        Args:
            model: PINN模型
            collocation_points: 配置点 [N_colloc, input_dim]
            params: 物理参数字典
        
        Returns:
            residual_loss: 物理残差损失
        """
        # 计算物理残差
        residual = model.compute_physics_residual(collocation_points, params)
        
        # 检查NaN值
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            print(f"Warning: NaN or Inf detected in residual, shape: {residual.shape}")
            residual = torch.where(torch.isnan(residual) | torch.isinf(residual), 
                                 torch.tensor(0.0, device=residual.device), residual)
        
        # MSE损失
        loss = self.mse_loss(residual, torch.zeros_like(residual))
        
        return loss
    
    def boundary_loss(self, model, boundary_points: torch.Tensor, 
                     boundary_values: torch.Tensor) -> torch.Tensor:
        """边界条件损失 L_boundary
        
        Args:
            model: PINN模型
            boundary_points: 边界点 [N_bc, input_dim]
            boundary_values: 边界值 [N_bc, 1]
        
        Returns:
            boundary_loss: 边界条件损失
        """
        # 模型预测
        pred_values = model(boundary_points)
        
        # 检查NaN值
        if torch.isnan(pred_values).any() or torch.isinf(pred_values).any():
            print(f"Warning: NaN or Inf detected in boundary prediction, shape: {pred_values.shape}")
            pred_values = torch.where(torch.isnan(pred_values) | torch.isinf(pred_values), 
                                    torch.tensor(0.0, device=pred_values.device), pred_values)
        
        # MSE损失
        loss = self.mse_loss(pred_values, boundary_values)
        
        return loss
    
    def initial_loss(self, model, initial_points: torch.Tensor, 
                    initial_values: torch.Tensor) -> torch.Tensor:
        """初始条件损失 L_initial
        
        Args:
            model: PINN模型
            initial_points: 初始点 [N_ic, input_dim]
            initial_values: 初始值 [N_ic, 1]
        
        Returns:
            initial_loss: 初始条件损失
        """
        # 模型预测
        pred_values = model(initial_points)
        
        # 检查NaN值
        if torch.isnan(pred_values).any() or torch.isinf(pred_values).any():
            print(f"Warning: NaN or Inf detected in initial prediction, shape: {pred_values.shape}")
            pred_values = torch.where(torch.isnan(pred_values) | torch.isinf(pred_values), 
                                    torch.tensor(0.0, device=pred_values.device), pred_values)
        
        # MSE损失
        loss = self.mse_loss(pred_values, initial_values)
        
        return loss
    
    def data_loss(self, model, data_points: torch.Tensor, 
                 data_values: torch.Tensor) -> torch.Tensor:
        """数据损失 L_data
        
        Args:
            model: PINN模型
            data_points: 数据点 [N_data, input_dim]
            data_values: 数据值 [N_data, 1]
        
        Returns:
            data_loss: 数据损失
        """
        # 模型预测
        pred_values = model(data_points)
        
        # 检查NaN值
        if torch.isnan(pred_values).any() or torch.isinf(pred_values).any():
            print(f"Warning: NaN or Inf detected in data prediction, shape: {pred_values.shape}")
            pred_values = torch.where(torch.isnan(pred_values) | torch.isinf(pred_values), 
                                    torch.tensor(0.0, device=pred_values.device), pred_values)
        
        # MSE损失
        loss = self.mse_loss(pred_values, data_values)
        
        return loss
    
    def forward(self, model, data_dict: Dict[str, torch.Tensor], 
               params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总损失
        
        Args:
            model: PINN模型
            data_dict: 包含各种数据的字典
                - 'collocation_points': 配置点
                - 'boundary_points': 边界点
                - 'boundary_values': 边界值
                - 'initial_points': 初始点
                - 'initial_values': 初始值
                - 'data_points': 数据点
                - 'data_values': 数据值
            params: 物理参数字典
        
        Returns:
            total_loss: 总损失
            loss_components: 各损失分量字典
        """
        # 初始化损失分量
        self.loss_components = {}
        
        # 物理残差损失
        if 'collocation_points' in data_dict:
            l_residual = self.residual_loss(
                model, data_dict['collocation_points'], params
            )
            self.loss_components['residual'] = l_residual
        else:
            self.loss_components['residual'] = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # 边界条件损失
        if 'boundary_points' in data_dict and 'boundary_values' in data_dict:
            l_boundary = self.boundary_loss(
                model, data_dict['boundary_points'], data_dict['boundary_values']
            )
            self.loss_components['boundary'] = l_boundary
        else:
            self.loss_components['boundary'] = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # 初始条件损失
        if 'initial_points' in data_dict and 'initial_values' in data_dict:
            l_initial = self.initial_loss(
                model, data_dict['initial_points'], data_dict['initial_values']
            )
            self.loss_components['initial'] = l_initial
        else:
            self.loss_components['initial'] = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # 数据损失
        if 'data_points' in data_dict and 'data_values' in data_dict:
            l_data = self.data_loss(
                model, data_dict['data_points'], data_dict['data_values']
            )
            self.loss_components['data'] = l_data
        else:
            self.loss_components['data'] = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # 检查各个损失分量中的NaN值
        for name, loss in self.loss_components.items():
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf detected in {name} loss")
                self.loss_components[name] = torch.where(torch.isnan(loss) | torch.isinf(loss),
                                                       torch.tensor(0.0, device=loss.device), loss)
        
        # 计算总损失
        total_loss = (
            self.weights['residual'] * self.loss_components['residual'] +
            self.weights['boundary'] * self.loss_components['boundary'] +
            self.weights['initial'] * self.loss_components['initial'] +
            self.weights['data'] * self.loss_components['data']
        )
        
        # 检查总损失中的NaN值
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("Warning: NaN or Inf detected in total loss")
            total_loss = torch.where(torch.isnan(total_loss) | torch.isinf(total_loss),
                                   torch.tensor(1e6, device=total_loss.device), total_loss)
        
        return total_loss, self.loss_components
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新损失权重"""
        self.weights.update(new_weights)
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

class AdaptiveWeighting:
    """自适应权重调整策略"""
    
    def __init__(self, method: str = 'gradient_normalization', alpha: float = 0.9):
        """
        Args:
            method: 权重调整方法 ('gradient_normalization', 'loss_balancing')
            alpha: 指数移动平均系数
        """
        self.method = method
        self.alpha = alpha
        self.grad_norms_history = {}
        
    def compute_gradient_norms(self, loss_components: Dict[str, torch.Tensor], 
                              model_parameters) -> Dict[str, float]:
        """计算各损失分量的梯度范数"""
        grad_norms = {}
        
        for loss_name, loss_value in loss_components.items():
            if loss_value.requires_grad:
                # 计算梯度
                try:
                    grads = torch.autograd.grad(
                        outputs=loss_value,
                        inputs=model_parameters,
                        retain_graph=True,
                        allow_unused=True
                    )
                    
                    # 计算梯度范数
                    grad_norm = 0.0
                    for grad in grads:
                        if grad is not None:
                            grad_norm += grad.norm().item() ** 2
                    grad_norms[loss_name] = grad_norm ** 0.5
                except Exception as e:
                    print(f"Error computing gradient for {loss_name}: {e}")
                    grad_norms[loss_name] = 0.0
            else:
                grad_norms[loss_name] = 0.0
        
        return grad_norms
    
    def update_weights_gradient_normalization(self, loss_components: Dict[str, torch.Tensor],
                                            model_parameters, current_weights: Dict[str, float]) -> Dict[str, float]:
        """基于梯度归一化的权重更新"""
        # 计算梯度范数
        grad_norms = self.compute_gradient_norms(loss_components, model_parameters)
        
        # 更新历史记录
        for loss_name, grad_norm in grad_norms.items():
            if loss_name not in self.grad_norms_history:
                self.grad_norms_history[loss_name] = grad_norm
            else:
                self.grad_norms_history[loss_name] = (
                    self.alpha * self.grad_norms_history[loss_name] + 
                    (1 - self.alpha) * grad_norm
                )
        
        # 计算新权重
        new_weights = {}
        max_grad_norm = max(self.grad_norms_history.values()) + 1e-8
        
        for loss_name in current_weights.keys():
            if loss_name in self.grad_norms_history:
                new_weights[loss_name] = max_grad_norm / (self.grad_norms_history[loss_name] + 1e-8)
            else:
                new_weights[loss_name] = current_weights[loss_name]
        
        return new_weights
    
    def update_weights_loss_balancing(self, loss_components: Dict[str, torch.Tensor],
                                    current_weights: Dict[str, float]) -> Dict[str, float]:
        """基于损失平衡的权重更新"""
        new_weights = {}
        
        # 计算损失值
        loss_values = {name: loss.item() for name, loss in loss_components.items()}
        max_loss = max(loss_values.values()) + 1e-8
        
        for loss_name in current_weights.keys():
            if loss_name in loss_values:
                new_weights[loss_name] = max_loss / (loss_values[loss_name] + 1e-8)
            else:
                new_weights[loss_name] = current_weights[loss_name]
        
        return new_weights
    
    def update_weights(self, loss_components: Dict[str, torch.Tensor],
                      model_parameters, current_weights: Dict[str, float]) -> Dict[str, float]:
        """更新权重"""
        if self.method == 'gradient_normalization':
            return self.update_weights_gradient_normalization(
                loss_components, model_parameters, current_weights
            )
        elif self.method == 'loss_balancing':
            return self.update_weights_loss_balancing(loss_components, current_weights)
        else:
            return current_weights