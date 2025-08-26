"""神经网络架构模块

定义航天器表面充电PINN的神经网络架构：
1. 主网络 V_NN: 预测表面电位 V(t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id)
2. 子网络 NN_yield: 预测电子发射电流密度 J_emission(V, material_id)

使用DeepXDE框架构建，支持自动微分和物理约束。

Author: PINN Engineering Team
"""

import torch
import torch.nn as nn
import deepxde as dde
import numpy as np
from typing import Dict, List, Tuple, Optional


class ActivationFactory:
    """激活函数工厂类"""
    
    @staticmethod
    def get_activation(name: str) -> nn.Module:
        """获取激活函数
        
        Args:
            name: 激活函数名称 ('tanh', 'swish', 'silu', 'gelu')
            
        Returns:
            activation: PyTorch激活函数
        """
        activations = {
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),  # Swish = SiLU
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unsupported activation function: {name}. "
                           f"Supported: {list(activations.keys())}")
        
        return activations[name.lower()]


class MainNetwork(nn.Module):
    """主网络 V_NN
    
    输入: [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
    输出: 表面电位 V
    """
    
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        """初始化主网络
        
        Args:
            layers: 网络层配置，例如 [8, 64, 64, 64, 1]
            activation: 激活函数名称
        """
        super(MainNetwork, self).__init__()
        
        if len(layers) < 2:
            raise ValueError("Network must have at least input and output layers")
        
        self.layers = layers
        self.activation_name = activation
        self.activation = ActivationFactory.get_activation(activation)
        
        # 构建网络层
        self.network_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.network_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重
        
        使用Xavier初始化确保梯度稳定性。
        """
        for layer in self.network_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, 8]
               [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
        
        Returns:
            V: 表面电位 [batch_size, 1]
        """
        # 输入归一化（可选）
        # x = self._normalize_input(x)
        
        # 前向传播
        for i, layer in enumerate(self.network_layers):
            x = layer(x)
            # 除了最后一层，都应用激活函数
            if i < len(self.network_layers) - 1:
                x = self.activation(x)
        
        return x
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化（可选实现）
        
        Args:
            x: 原始输入
            
        Returns:
            normalized_x: 归一化后的输入
        """
        # 这里可以实现输入归一化逻辑
        # 例如：对数尺度归一化、标准化等
        return x


class YieldNetwork(nn.Module):
    """子网络 NN_yield
    
    输入: [V, material_id]
    输出: 总电子发射电流密度 J_emission
    """
    
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        """初始化子网络
        
        Args:
            layers: 网络层配置，例如 [2, 32, 32, 1]
            activation: 激活函数名称
        """
        super(YieldNetwork, self).__init__()
        
        if len(layers) < 2:
            raise ValueError("Network must have at least input and output layers")
        
        self.layers = layers
        self.activation_name = activation
        self.activation = ActivationFactory.get_activation(activation)
        
        # 构建网络层
        self.network_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.network_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in self.network_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, 2] - [V, material_id]
        
        Returns:
            J_emission: 电子发射电流密度 [batch_size, 1]
        """
        # 前向传播
        for i, layer in enumerate(self.network_layers):
            x = layer(x)
            # 除了最后一层，都应用激活函数
            if i < len(self.network_layers) - 1:
                x = self.activation(x)
        
        # 确保输出为非负值（电子发射电流应为正值）
        # 使用softplus确保输出始终为正
        x = torch.nn.functional.softplus(x)
        
        return x


class CombinedPINNModel:
    """组合PINN模型
    
    整合主网络和子网络，构建完整的灰箱PINN模型。
    """
    
    def __init__(self, config: Dict):
        """初始化组合模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 网络配置
        main_config = config['network']['main_network']
        yield_config = config['network']['yield_network']
        
        # 创建网络
        self.main_net = MainNetwork(
            layers=main_config['layers'],
            activation=main_config['activation']
        )
        
        self.yield_net = YieldNetwork(
            layers=yield_config['layers'],
            activation=yield_config['activation']
        )
        
        # 输入输出维度
        self.input_dim = main_config['layers'][0]  # 8: [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
        self.output_dim = 1  # V
    
    def predict_potential(self, inputs: torch.Tensor) -> torch.Tensor:
        """预测表面电位
        
        Args:
            inputs: 输入参数 [batch_size, 8]
        
        Returns:
            V: 表面电位 [batch_size, 1]
        """
        return self.main_net(inputs)
    
    def predict_emission_current(self, V: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """预测电子发射电流
        
        Args:
            V: 表面电位 [batch_size, 1]
            material_id: 材料标识 [batch_size, 1]
        
        Returns:
            J_emission: 电子发射电流密度 [batch_size, 1]
        """
        yield_input = torch.cat([V, material_id], dim=1)
        return self.yield_net(yield_input)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整前向传播
        
        Args:
            inputs: 输入参数 [batch_size, 8]
        
        Returns:
            V: 表面电位 [batch_size, 1]
            J_emission: 电子发射电流密度 [batch_size, 1]
        """
        # 预测电位
        V = self.predict_potential(inputs)
        
        # 提取材料标识
        material_id = inputs[:, -1:]  # 最后一列是material_id
        
        # 预测发射电流
        J_emission = self.predict_emission_current(V, material_id)
        
        return V, J_emission
    
    def get_networks(self) -> Tuple[nn.Module, nn.Module]:
        """获取网络实例
        
        Returns:
            main_net: 主网络
            yield_net: 子网络
        """
        return self.main_net, self.yield_net
    
    def get_network_info(self) -> Dict:
        """获取网络信息
        
        Returns:
            info: 网络架构信息
        """
        return {
            'main_network': {
                'layers': self.main_net.layers,
                'activation': self.main_net.activation_name,
                'parameters': sum(p.numel() for p in self.main_net.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.main_net.parameters() if p.requires_grad)
            },
            'yield_network': {
                'layers': self.yield_net.layers,
                'activation': self.yield_net.activation_name,
                'parameters': sum(p.numel() for p in self.yield_net.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.yield_net.parameters() if p.requires_grad)
            },
            'total_parameters': sum(p.numel() for p in self.main_net.parameters()) + 
                              sum(p.numel() for p in self.yield_net.parameters())
        }


class DeepXDENetworkWrapper:
    """DeepXDE网络包装器
    
    将PyTorch网络包装为DeepXDE兼容的格式。
    """
    
    def __init__(self, combined_model: CombinedPINNModel):
        """初始化包装器
        
        Args:
            combined_model: 组合PINN模型
        """
        self.combined_model = combined_model
        self.main_net, self.yield_net = combined_model.get_networks()
    
    def create_deepxde_net(self, input_transform=None, output_transform=None):
        """创建DeepXDE网络
        
        Args:
            input_transform: 输入变换函数
            output_transform: 输出变换函数
            
        Returns:
            net: DeepXDE网络实例
        """
        # 创建DeepXDE全连接网络
        main_config = self.combined_model.config['network']['main_network']
        
        net = dde.nn.FNN(
            layer_sizes=main_config['layers'],
            activation=main_config['activation'],
            kernel_initializer="Glorot uniform"
        )
        
        # 应用变换（如果提供）
        if input_transform is not None:
            net.apply_feature_transform(input_transform)
        
        if output_transform is not None:
            net.apply_output_transform(output_transform)
        
        return net


def create_networks(config: Dict) -> CombinedPINNModel:
    """创建网络的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        combined_model: 组合PINN模型
    """
    return CombinedPINNModel(config)


def print_network_summary(combined_model: CombinedPINNModel):
    """打印网络摘要信息
    
    Args:
        combined_model: 组合PINN模型
    """
    info = combined_model.get_network_info()
    
    print("=" * 60)
    print("PINN Network Architecture Summary")
    print("=" * 60)
    
    print("\nMain Network (V_NN):")
    print(f"  Layers: {info['main_network']['layers']}")
    print(f"  Activation: {info['main_network']['activation']}")
    print(f"  Parameters: {info['main_network']['parameters']:,}")
    
    print("\nYield Network (NN_yield):")
    print(f"  Layers: {info['yield_network']['layers']}")
    print(f"  Activation: {info['yield_network']['activation']}")
    print(f"  Parameters: {info['yield_network']['parameters']:,}")
    
    print(f"\nTotal Parameters: {info['total_parameters']:,}")
    print("=" * 60)