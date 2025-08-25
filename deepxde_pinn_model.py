import numpy as np
import torch
import torch.nn as nn
import deepxde as dde
from typing import Dict, List, Tuple, Optional, Callable

class YieldNetwork(nn.Module):
    """子网络NNyield用于学习电子激发发射电流密度
    
    这个网络学习总的电子激发发射电流密度Jemission = Jse + Jbse
    替代传统的复杂积分和经验公式
    """
    
    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = [64, 64, 32], 
                 activation: str = 'tanh'):
        super(YieldNetwork, self).__init__()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: [V, Je_inc, material_prop, additional_features]
                   形状: (batch_size, input_dim)
        
        Returns:
            Jemission: 电子激发发射电流密度 (batch_size, 1)
        """
        return torch.abs(self.network(inputs))  # 确保电流密度为正

class SurfaceChargingPDE:
    """表面充电物理模型和PDE定义"""
    
    def __init__(self, yield_network: YieldNetwork, 
                 physical_params: Dict[str, float]):
        self.yield_network = yield_network
        self.params = physical_params
        
        # 物理常数
        self.k_B = 1.380649e-23  # 玻尔兹曼常数 [J/K]
        self.e = 1.602176634e-19  # 电子电荷 [C]
    
    def plasma_current_electron(self, V: torch.Tensor, 
                              n_e: torch.Tensor, T_e: torch.Tensor) -> torch.Tensor:
        """入射等离子体电子电流 - 轨道限制理论"""
        # 无电位表面的随机热电流密度
        J_e0 = 0.25 * n_e * self.e * torch.sqrt(8 * self.k_B * T_e / (np.pi * 9.109e-31))
        
        # 电位相关项
        potential_term = -self.e * V / (self.k_B * T_e)
        
        # 吸引性电位 (V < 0)
        attractive_mask = V < 0
        J_attractive = J_e0 * (1 - potential_term)
        
        # 排斥性电位 (V > 0)
        repulsive_mask = V >= 0
        J_repulsive = J_e0 * torch.exp(potential_term)
        
        return torch.where(attractive_mask, J_attractive, J_repulsive)
    
    def plasma_current_ion(self, V: torch.Tensor, 
                          n_i: torch.Tensor, T_i: torch.Tensor) -> torch.Tensor:
        """入射等离子体离子电流 - 轨道限制理论"""
        # 假设离子质量为质子质量
        m_i = 1.673e-27  # kg
        J_i0 = 0.25 * n_i * self.e * torch.sqrt(8 * self.k_B * T_i / (np.pi * m_i))
        
        potential_term = self.e * V / (self.k_B * T_i)
        
        # 对离子，电位符号相反
        attractive_mask = V > 0
        J_attractive = J_i0 * (1 + potential_term)
        
        repulsive_mask = V <= 0
        J_repulsive = J_i0 * torch.exp(-potential_term)
        
        return torch.where(attractive_mask, J_attractive, J_repulsive)
    
    def photoelectron_current(self, V: torch.Tensor, 
                            S_flux: torch.Tensor) -> torch.Tensor:
        """光电子电流 - 自限模型"""
        # 饱和光电子电流密度 (与太阳辐射通量成正比)
        J_ph0 = self.params.get('photoelectron_yield', 1e-5) * S_flux
        
        # 等效光电子温度
        V_ph = self.params.get('photoelectron_temperature', 2.0)  # eV
        
        # 自限效应
        limiting_mask = V > 0
        J_limited = J_ph0 * torch.exp(-V / V_ph)
        
        return torch.where(limiting_mask, J_limited, J_ph0)
    
    def net_current_density(self, inputs: torch.Tensor, 
                          V: torch.Tensor) -> torch.Tensor:
        """计算净电流密度
        
        Args:
            inputs: 包含所有输入参数的张量
            V: 表面电位
        
        Returns:
            J_net: 净电流密度
        """
        # 解析输入参数
        # inputs格式: [t, x, y, z, n_e, T_e, n_i, T_i, S_flux, alpha_sun, alpha_ram, material_id]
        n_e = inputs[:, 4:5]
        T_e = inputs[:, 5:6]
        n_i = inputs[:, 6:7]
        T_i = inputs[:, 7:8]
        S_flux = inputs[:, 8:9]
        
        # 计算各电流分量
        J_e = self.plasma_current_electron(V, n_e, T_e)
        J_i = self.plasma_current_ion(V, n_i, T_i)
        J_ph = self.photoelectron_current(V, S_flux)
        
        # 使用子网络计算电子激发发射电流
        # 子网络输入: [V, J_e, material_prop, additional_features]
        material_id = inputs[:, -1:]
        yield_input = torch.cat([V, J_e, material_id, S_flux], dim=1)
        J_emission = self.yield_network(yield_input)
        
        # 净电流密度 (注意符号约定)
        J_net = J_e - J_i - J_emission - J_ph
        
        return J_net
    
    def pde_residual(self, inputs: torch.Tensor, 
                    outputs: torch.Tensor) -> torch.Tensor:
        """PDE残差函数
        
        控制方程: C(x) * ∂V/∂t = J_net(V, E_env, M_prop, x, t)
        
        Args:
            inputs: 输入张量 [t, x, y, z, ...]
            outputs: 网络输出 V(inputs)
        
        Returns:
            residual: PDE残差
        """
        V = outputs
        
        # 计算时间导数 ∂V/∂t
        dV_dt = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        
        # 单位面积电容 (可以是位置相关的)
        C = self.params.get('capacitance', 1e-12)  # F/m^2
        
        # 计算净电流密度
        J_net = self.net_current_density(inputs, V)
        
        # PDE残差
        residual = C * dV_dt - J_net
        
        return residual

class DeepXDESurfaceChargingPINN:
    """基于DeepXDE的表面充电PINN模型"""
    
    def __init__(self, 
                 domain_bounds: Dict[str, Tuple[float, float]],
                 physical_params: Dict[str, float],
                 network_config: Dict[str, any]):
        
        self.domain_bounds = domain_bounds
        self.physical_params = physical_params
        self.network_config = network_config
        
        # 创建子网络
        self.yield_network = YieldNetwork(
            input_dim=network_config.get('yield_input_dim', 4),
            hidden_dims=network_config.get('yield_hidden_dims', [64, 64, 32]),
            activation=network_config.get('activation', 'tanh')
        )
        
        # 创建物理模型
        self.physics = SurfaceChargingPDE(self.yield_network, physical_params)
        
        # 定义计算域
        self._setup_domain()
        
        # 创建主网络
        self._setup_main_network()
        
        # 创建PINN模型
        self._setup_pinn_model()
    
    def _setup_domain(self):
        """设置计算域"""
        # 时空域
        t_bounds = self.domain_bounds['t']
        x_bounds = self.domain_bounds['x']
        y_bounds = self.domain_bounds.get('y', (0, 1))
        z_bounds = self.domain_bounds.get('z', (0, 1))
        
        # 环境参数域
        n_e_bounds = self.domain_bounds.get('n_e', (1e6, 1e12))
        T_e_bounds = self.domain_bounds.get('T_e', (1000, 50000))
        n_i_bounds = self.domain_bounds.get('n_i', (1e6, 1e12))
        T_i_bounds = self.domain_bounds.get('T_i', (300, 10000))
        S_flux_bounds = self.domain_bounds.get('S_flux', (0, 1400))
        alpha_sun_bounds = self.domain_bounds.get('alpha_sun', (0, np.pi/2))
        alpha_ram_bounds = self.domain_bounds.get('alpha_ram', (0, 2*np.pi))
        material_bounds = self.domain_bounds.get('material_id', (0, 5))
        
        # 创建几何域
        self.geom = dde.geometry.Hypercube(
            [t_bounds[0], x_bounds[0], y_bounds[0], z_bounds[0],
             n_e_bounds[0], T_e_bounds[0], n_i_bounds[0], T_i_bounds[0],
             S_flux_bounds[0], alpha_sun_bounds[0], alpha_ram_bounds[0], material_bounds[0]],
            [t_bounds[1], x_bounds[1], y_bounds[1], z_bounds[1],
             n_e_bounds[1], T_e_bounds[1], n_i_bounds[1], T_i_bounds[1],
             S_flux_bounds[1], alpha_sun_bounds[1], alpha_ram_bounds[1], material_bounds[1]]
        )
    
    def _setup_main_network(self):
        """设置主神经网络VNN"""
        # 网络架构配置
        layer_sizes = [12] + self.network_config.get('hidden_layers', [128, 128, 128, 128]) + [1]
        activation = self.network_config.get('activation', 'tanh')
        initializer = self.network_config.get('initializer', 'Glorot uniform')
        
        # 创建前馈神经网络
        self.main_network = dde.nn.FNN(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_initializer=initializer
        )
    
    def _setup_pinn_model(self):
        """设置PINN模型"""
        # 创建PDE
        self.pde = dde.PDE(
            self.geom,
            self.physics.pde_residual,
            [],  # 边界条件将单独添加
            num_domain=self.network_config.get('num_domain', 10000),
            num_boundary=self.network_config.get('num_boundary', 1000),
            num_test=self.network_config.get('num_test', 1000)
        )
        
        # 创建模型
        self.model = dde.Model(self.pde, self.main_network)
    
    def add_boundary_conditions(self, boundary_conditions: List[Dict]):
        """添加边界条件
        
        Args:
            boundary_conditions: 边界条件列表，每个元素包含:
                - type: 'dirichlet' 或 'neumann'
                - boundary: 边界函数
                - value: 边界值函数
        """
        bcs = []
        
        for bc_config in boundary_conditions:
            if bc_config['type'] == 'dirichlet':
                bc = dde.DirichletBC(
                    self.geom,
                    bc_config['value'],
                    bc_config['boundary']
                )
            elif bc_config['type'] == 'neumann':
                bc = dde.NeumannBC(
                    self.geom,
                    bc_config['value'],
                    bc_config['boundary']
                )
            elif bc_config['type'] == 'initial':
                bc = dde.IC(
                    self.geom,
                    bc_config['value'],
                    bc_config['boundary']
                )
            
            bcs.append(bc)
        
        # 更新PDE
        self.pde = dde.PDE(
            self.geom,
            self.physics.pde_residual,
            bcs,
            num_domain=self.network_config.get('num_domain', 10000),
            num_boundary=self.network_config.get('num_boundary', 1000),
            num_test=self.network_config.get('num_test', 1000)
        )
        
        # 重新创建模型
        self.model = dde.Model(self.pde, self.main_network)
    
    def add_data_constraints(self, X_data: np.ndarray, y_data: np.ndarray):
        """添加数据约束
        
        Args:
            X_data: 输入数据点
            y_data: 对应的输出数据
        """
        # 创建PointSetBC用于数据约束
        data_bc = dde.PointSetBC(X_data, y_data)
        
        # 获取现有边界条件
        existing_bcs = self.pde.bcs if hasattr(self.pde, 'bcs') else []
        existing_bcs.append(data_bc)
        
        # 更新PDE
        self.pde = dde.PDE(
            self.geom,
            self.physics.pde_residual,
            existing_bcs,
            num_domain=self.network_config.get('num_domain', 10000),
            num_boundary=self.network_config.get('num_boundary', 1000),
            num_test=self.network_config.get('num_test', 1000)
        )
        
        # 重新创建模型
        self.model = dde.Model(self.pde, self.main_network)
    
    def compile_model(self, optimizer_config: Dict[str, any]):
        """编译模型
        
        Args:
            optimizer_config: 优化器配置
        """
        # 设置优化器
        if optimizer_config.get('type', 'adam') == 'adam':
            optimizer = 'adam'
        elif optimizer_config.get('type') == 'lbfgs':
            optimizer = 'L-BFGS'
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            lr=optimizer_config.get('learning_rate', 1e-3),
            loss_weights=optimizer_config.get('loss_weights', None)
        )
    
    def train(self, iterations: int, 
              display_every: int = 1000,
              save_path: Optional[str] = None) -> dde.Model:
        """训练模型
        
        Args:
            iterations: 训练迭代次数
            display_every: 显示频率
            save_path: 模型保存路径
        
        Returns:
            训练历史
        """
        # 训练模型
        losshistory, train_state = self.model.train(
            iterations=iterations,
            display_every=display_every
        )
        
        # 保存模型
        if save_path:
            self.model.save(save_path)
        
        return losshistory, train_state
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 输入数据
        
        Returns:
            预测结果
        """
        return self.model.predict(X)
    
    def get_model(self) -> dde.Model:
        """获取DeepXDE模型"""
        return self.model
    
    def get_yield_network(self) -> YieldNetwork:
        """获取子网络"""
        return self.yield_network