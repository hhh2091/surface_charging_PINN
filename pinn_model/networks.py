# 神经网络架构模块
# 定义主网络和子网络的PyTorch架构

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np

class ActivationFunction:
    """
    激活函数工厂类
    """
    
    @staticmethod
    def get_activation(name: str) -> nn.Module:
        """
        根据名称获取激活函数
        
        Args:
            name: 激活函数名称 ('tanh', 'swish', 'silu', 'gelu')
            
        Returns:
            activation: 激活函数模块
        """
        name = name.lower()
        if name == 'tanh':
            return nn.Tanh()
        elif name in ['swish', 'silu']:
            return nn.SiLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

class MainNetwork(nn.Module):
    """
    主网络 V_NN
    
    输入: [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
    输出: 表面电位 V(p)
    
    使用全连接层和光滑激活函数（tanh或swish）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化主网络
        
        Args:
            config: 网络配置字典
        """
        super(MainNetwork, self).__init__()
        
        self.config = config
        network_config = config['network']['main_network']
        
        self.input_dim = network_config['input_dim']
        self.hidden_layers = network_config['hidden_layers']
        self.output_dim = network_config['output_dim']
        self.activation_name = network_config['activation']
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        
        # 隐藏层
        for i in range(len(self.hidden_layers) - 1):
            self.layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
        
        # 输出层
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
        
        # 激活函数
        self.activation = ActivationFunction.get_activation(self.activation_name)
        
        # 初始化权重
        self._initialize_weights()
        
        # 输入归一化参数（将在训练时设置）
        self.register_buffer('input_mean', torch.zeros(self.input_dim))
        self.register_buffer('input_std', torch.ones(self.input_dim))
        self.register_buffer('output_mean', torch.zeros(self.output_dim))
        self.register_buffer('output_std', torch.ones(self.output_dim))
        
    def _initialize_weights(self):
        """
        初始化网络权重
        使用Xavier初始化以确保梯度稳定性
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def set_normalization_params(self, input_stats: Tuple[torch.Tensor, torch.Tensor],
                               output_stats: Tuple[torch.Tensor, torch.Tensor]):
        """
        设置输入输出归一化参数
        
        Args:
            input_stats: (input_mean, input_std)
            output_stats: (output_mean, output_std)
        """
        self.input_mean.copy_(input_stats[0])
        self.input_std.copy_(input_stats[1])
        self.output_mean.copy_(output_stats[0])
        self.output_std.copy_(output_stats[1])
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化输入
        
        Args:
            x: 原始输入
            
        Returns:
            x_norm: 归一化后的输入
        """
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        反归一化输出
        
        Args:
            y: 归一化的输出
            
        Returns:
            y_denorm: 反归一化后的输出
        """
        return y * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            normalize: 是否进行输入归一化
            
        Returns:
            output: 输出张量 [batch_size, output_dim]
        """
        if normalize:
            x = self.normalize_input(x)
        
        # 前向传播通过所有隐藏层
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # 输出层（不使用激活函数）
        output = self.layers[-1](x)
        
        if normalize:
            output = self.denormalize_output(output)
        
        return output

class YieldNetwork(nn.Module):
    """
    子网络 NN_yield
    
    输入: [V, material_id]
    输出: 总的电子发射电流密度 J_emission
    
    用于建模不确定的电子发射过程
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化子网络
        
        Args:
            config: 网络配置字典
        """
        super(YieldNetwork, self).__init__()
        
        self.config = config
        network_config = config['network']['yield_network']
        
        self.input_dim = network_config['input_dim']
        self.hidden_layers = network_config['hidden_layers']
        self.output_dim = network_config['output_dim']
        self.activation_name = network_config['activation']
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        
        # 隐藏层
        for i in range(len(self.hidden_layers) - 1):
            self.layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
        
        # 输出层
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
        
        # 激活函数
        self.activation = ActivationFunction.get_activation(self.activation_name)
        
        
        # 材料嵌入层（可选）
        self.use_material_embedding = True
        if self.use_material_embedding:
            self.material_embedding = nn.Embedding(num_embeddings=2, embedding_dim=4)
            # 调整输入维度
            self.layers[0] = nn.Linear(1 + 4, self.hidden_layers[0])  # V + material_embedding
        # 初始化权重
        self._initialize_weights()
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        if self.use_material_embedding:
            nn.init.normal_(self.material_embedding.weight, mean=0, std=0.1)
    
    def forward(self, V: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            V: 表面电位 [batch_size, 1]
            material_id: 材料标识 [batch_size, 1]
            
        Returns:
            J_emission: 发射电流密度 [batch_size, 1]
        """
        if self.use_material_embedding:
            # 使用材料嵌入
            material_id_int = material_id.long().squeeze(-1)
            material_emb = self.material_embedding(material_id_int)
            x = torch.cat([V, material_emb], dim=-1)
        else:
            # 直接拼接
            x = torch.cat([V, material_id], dim=-1)
        
        # 前向传播通过所有隐藏层
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # 输出层
        output = self.layers[-1](x)
        
        # 确保输出为正值（电流密度应为正）
        output = torch.abs(output)
        
        return output

class PINN(nn.Module):
    """
    完整的参数化灰箱PINN模型
    
    结合主网络和子网络，实现端到端的航天器表面充电预测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PINN模型
        
        Args:
            config: 完整的配置字典
        """
        super(PINN, self).__init__()
        
        self.config = config
        
        # 初始化主网络和子网络
        self.main_network = MainNetwork(config)
        self.yield_network = YieldNetwork(config)
        
        # 物理模型（不包含可学习参数）
        from .physics import PhysicsModel
        self.physics_model = PhysicsModel(config)
        
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 输入张量 [batch_size, 8]
                   [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
            
        Returns:
            outputs: 包含所有输出的字典
                - V: 预测的表面电位
                - J_emission: 预测的发射电流密度
                - dV_dt: 电位时间导数（用于物理残差）
        """
        # 主网络预测表面电位
        V = self.main_network(inputs)
        
        # 提取材料标识
        material_id = inputs[:, 7:8]
        
        # 子网络预测发射电流密度
        J_emission = self.yield_network(V, material_id)
        
        # 计算电位的时间导数（用于物理残差）
        if inputs.requires_grad:
            dV_dt = torch.autograd.grad(
                outputs=V,
                inputs=inputs,
                grad_outputs=torch.ones_like(V),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]  # 只取时间导数
        else:
            dV_dt = torch.zeros_like(V)
        
        return {
            'V': V,
            'J_emission': J_emission,
            'dV_dt': dV_dt
        }
    
    def compute_physics_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算物理残差
        
        Args:
            inputs: 输入张量
            
        Returns:
            residual: 物理残差
        """
        outputs = self.forward(inputs)
        
        residual = self.physics_model.physics_residual(
            V=outputs['V'],
            dV_dt=outputs['dV_dt'],
            J_emission=outputs['J_emission'],
            inputs=inputs
        )
        
        return residual
    
    def get_current_components(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取所有电流分量
        
        Args:
            inputs: 输入张量
            
        Returns:
            current_components: 电流分量字典
        """
        outputs = self.forward(inputs)
        
        current_components = self.physics_model.get_current_components(
            V=outputs['V'],
            J_emission=outputs['J_emission'],
            inputs=inputs
        )
        
        return current_components
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        预测表面电位（推理模式）
        
        Args:
            inputs: 输入张量
            
        Returns:
            V: 预测的表面电位
        """
        with torch.no_grad():
            outputs = self.forward(inputs)
            return outputs['V']

def create_pinn_model(config: Dict[str, Any]) -> PINN:
    """
    创建PINN模型实例
    
    Args:
        config: 配置字典
        
    Returns:
        model: PINN模型实例
    """
    return PINN(config)

def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        num_params: 参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: PINN):
    """
    打印模型结构摘要
    
    Args:
        model: PINN模型
    """
    print("=" * 60)
    print("PINN Model Summary")
    print("=" * 60)
    
    print(f"Main Network:")
    print(f"  Input dim: {model.main_network.input_dim}")
    print(f"  Hidden layers: {model.main_network.hidden_layers}")
    print(f"  Output dim: {model.main_network.output_dim}")
    print(f"  Activation: {model.main_network.activation_name}")
    print(f"  Parameters: {count_parameters(model.main_network):,}")
    
    print(f"\nYield Network:")
    print(f"  Input dim: {model.yield_network.input_dim}")
    print(f"  Hidden layers: {model.yield_network.hidden_layers}")
    print(f"  Output dim: {model.yield_network.output_dim}")
    print(f"  Activation: {model.yield_network.activation_name}")
    print(f"  Parameters: {count_parameters(model.yield_network):,}")
    
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    print("=" * 60)