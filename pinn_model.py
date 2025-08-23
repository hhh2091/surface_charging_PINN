import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

class VNN(nn.Module):
    """主神经网络，用于预测表面电位V"""
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 64, 64, 64], 
                 output_dim: int = 1, activation: str = 'tanh'):
        super(VNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 选择激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Tanh()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            包含: [t, x, y, z, ne, Te, ni, Ti, Sflux, alpha_sun, alpha_ram, material_id]
        
        Returns:
            V: 表面电位 [batch_size, 1]
        """
        return self.network(x)

class NNyield(nn.Module):
    """子网络，用于学习电子激发发射电流密度"""
    
    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = [32, 32, 32], 
                 output_dim: int = 1, activation: str = 'relu'):
        super(NNyield, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 选择激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        # 输出层使用ReLU确保电流密度为正
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, V: torch.Tensor, Je_inc: torch.Tensor, 
                Mprop: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            V: 表面电位 [batch_size, 1]
            Je_inc: 入射电子电流密度 [batch_size, 1]
            Mprop: 材料属性 [batch_size, n_material_props]
        
        Returns:
            J_emission: 总电子激发发射电流密度 (Jse + Jbse) [batch_size, 1]
        """
        # 拼接输入特征
        x = torch.cat([V, Je_inc, Mprop], dim=1)
        return self.network(x)

class SurfaceChargingPINN(nn.Module):
    """表面充电PINN模型"""
    
    def __init__(self, config: Dict):
        super(SurfaceChargingPINN, self).__init__()
        
        self.config = config
        
        # 主网络和子网络
        self.vnn = VNN(
            input_dim=config.get('vnn_input_dim', 12),
            hidden_dims=config.get('vnn_hidden_dims', [64, 64, 64, 64]),
            activation=config.get('vnn_activation', 'tanh')
        )
        
        self.nn_yield = NNyield(
            input_dim=config.get('yield_input_dim', 4),
            hidden_dims=config.get('yield_hidden_dims', [32, 32, 32]),
            activation=config.get('yield_activation', 'relu')
        )
        
        # 物理常数
        self.kb = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.e = 1.602176634e-19  # 电子电荷 (C)
        
    def compute_plasma_current(self, V: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算等离子体电流 (Je, Ji)
        
        使用轨道限制理论模型
        """
        ne = params['ne']  # 电子密度
        Te = params['Te']  # 电子温度
        ni = params['ni']  # 离子密度
        Ti = params['Ti']  # 离子温度
        
        # 电子随机热电流密度
        Je0 = 0.25 * ne * torch.sqrt(8 * self.kb * Te / (np.pi * 9.109e-31))  # me = 9.109e-31 kg
        
        # 离子随机热电流密度 (假设质子)
        Ji0 = 0.25 * ni * torch.sqrt(8 * self.kb * Ti / (np.pi * 1.673e-27))  # mp = 1.673e-27 kg
        
        # 电子电流 (对电子，qs = -e)
        V_norm_e = -self.e * V / (self.kb * Te)
        Je = torch.where(V < 0, 
                        Je0 * (1 - V_norm_e),  # 吸引性电位
                        Je0 * torch.exp(V_norm_e))  # 排斥性电位
        
        # 离子电流 (对离子，qs = +e)
        V_norm_i = self.e * V / (self.kb * Ti)
        Ji = torch.where(V > 0,
                        Ji0 * (1 - V_norm_i),  # 吸引性电位
                        Ji0 * torch.exp(V_norm_i))  # 排斥性电位
        
        return Je, Ji
    
    def compute_photoelectron_current(self, V: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算光电子电流 Jph"""
        Jph0 = params['Jph0']  # 饱和光电子电流密度
        Vph = params['Vph']    # 等效光电子温度
        
        Jph = torch.where(V <= 0,
                         Jph0,  # V <= 0
                         Jph0 * torch.exp(-V / Vph))  # V > 0
        
        return Jph
    
    def compute_net_current(self, V: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算净电流密度 Jnet"""
        # 等离子体电流
        Je, Ji = self.compute_plasma_current(V, params)
        
        # 光电子电流
        Jph = self.compute_photoelectron_current(V, params)
        
        # 电子激发发射电流 (通过子网络)
        Mprop = params['Mprop']  # 材料属性
        J_emission = self.nn_yield(V, Je, Mprop)
        
        # 净电流密度
        Jnet = Je - Ji - J_emission - Jph
        
        return Jnet
    
    def compute_physics_residual(self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算物理残差 f = C * dV/dt - Jnet"""
        # 启用梯度计算
        inputs.requires_grad_(True)
        
        # 前向传播得到V
        V = self.vnn(inputs)
        
        # 计算时间导数 dV/dt
        t = inputs[:, 0:1]  # 时间是第一个输入
        dV_dt = torch.autograd.grad(
            outputs=V, inputs=inputs,
            grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]  # 只取对时间的导数
        
        # 单位面积电容
        C = params['C']
        
        # 净电流密度
        Jnet = self.compute_net_current(V, params)
        
        # 物理残差
        residual = C * dV_dt - Jnet
        
        return residual
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.vnn(inputs)