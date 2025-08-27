# 物理方程模块
# 定义所有已知的物理方程（电流模型）

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class PhysicsModel:
    """
    航天器表面充电物理模型
    
    实现所有已知的电流模型:
    1. 入射等离子体电流 (J_e, J_i) - 基于轨道限制理论
    2. 光电子电流 (J_ph) - 受正电位抑制的饱和模型
    3. 控制方程 - 瞬态电流平衡ODE
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化物理模型
        
        Args:
            config: 配置字典，包含物理常数和材料参数
        """
        self.config = config
        self.physics_config = config['physics']
        
        # 基本物理常数
        self.e = float(self.physics_config['e'])  # 电子电荷 (C)
        self.k_B = float(self.physics_config['k_B'])  # 玻尔兹曼常数 (J/K)
        self.k_B_eV = float(self.physics_config['k_B_eV'])  # 玻尔兹曼常数 (eV/K)
        self.m_e = float(self.physics_config['m_e'])  # 电子质量 (kg)
        self.m_i = float(self.physics_config['m_i'])  # 质子质量 (kg)
        
        # 航天器参数
        self.A = float(self.physics_config['spacecraft']['area'])  # 表面积 (m^2)
        self.C = float(self.physics_config['spacecraft']['capacitance'])  # 等效电容 (F)
        self.T_ph = float(self.physics_config['spacecraft']['T_ph'])  # 光电子特征温度 (eV)
        
        # 材料参数
        self.materials = self.physics_config['materials']
        
    def thermal_current_density(self, n: torch.Tensor, T: torch.Tensor, 
                              mass: float, charge: float) -> torch.Tensor:
        """
        计算无电位表面的随机热电流密度 J_s0
        
        Args:
            n: 粒子密度 (m^-3)
            T: 粒子温度 (eV)
            mass: 粒子质量 (kg)
            charge: 粒子电荷 (C)
            
        Returns:
            J_s0: 热电流密度 (A/m^2)
        """
        # 将温度从eV转换为K
        T_kelvin = T / self.k_B_eV
        
        # J_s0 = n * |q_s| * sqrt(k_B * T_s / (2 * pi * m_s))
        thermal_velocity = torch.sqrt(self.k_B * T_kelvin / (2 * np.pi * mass))
        J_s0 = n * abs(charge) * thermal_velocity
        
        return J_s0
    
    def electron_current(self, V: torch.Tensor, n_e: torch.Tensor, 
                        T_e: torch.Tensor) -> torch.Tensor:
        """
        计算电子电流密度 J_e
        
        基于轨道限制理论:
        - 吸引性电位 (eV < 0): J_e = J_e0 * (1 - eV/(k_B*T_e))
        - 排斥性电位 (eV > 0): J_e = J_e0 * exp(eV/(k_B*T_e))
        
        Args:
            V: 表面电位 (V)
            n_e: 电子密度 (m^-3)
            T_e: 电子温度 (eV)
            
        Returns:
            J_e: 电子电流密度 (A/m^2)
        """
        # 计算热电流密度
        J_e0 = self.thermal_current_density(n_e, T_e, self.m_e, -self.e)
        
        # 计算无量纲电位 eV/(k_B*T_e)
        dimensionless_potential = self.e * V / (self.k_B_eV * T_e)
        
        # 根据电位符号选择公式
        attractive_mask = dimensionless_potential < 0  # eV < 0 (吸引电子)
        repulsive_mask = dimensionless_potential >= 0  # eV >= 0 (排斥电子)
        
        J_e = torch.zeros_like(V)
        
        # 吸引性电位: J_e = J_e0 * (1 - eV/(k_B*T_e))
        if attractive_mask.any():
            J_e[attractive_mask] = J_e0[attractive_mask] * (
                1 - dimensionless_potential[attractive_mask]
            )
        
        # 排斥性电位: J_e = J_e0 * exp(eV/(k_B*T_e))
        if repulsive_mask.any():
            J_e[repulsive_mask] = J_e0[repulsive_mask] * torch.exp(
                dimensionless_potential[repulsive_mask]
            )
        
        return J_e
    
    def ion_current(self, V: torch.Tensor, n_i: torch.Tensor, 
                   T_i: torch.Tensor) -> torch.Tensor:
        """
        计算离子电流密度 J_i
        
        基于轨道限制理论:
        - 吸引性电位 (eV > 0): J_i = J_i0 * (1 + eV/(k_B*T_i))
        - 排斥性电位 (eV < 0): J_i = J_i0 * exp(-eV/(k_B*T_i))
        
        Args:
            V: 表面电位 (V)
            n_i: 离子密度 (m^-3)
            T_i: 离子温度 (eV)
            
        Returns:
            J_i: 离子电流密度 (A/m^2)
        """
        # 计算热电流密度
        J_i0 = self.thermal_current_density(n_i, T_i, self.m_i, self.e)
        
        # 计算无量纲电位 eV/(k_B*T_i)
        dimensionless_potential = self.e * V / (self.k_B_eV * T_i)
        
        # 根据电位符号选择公式
        attractive_mask = dimensionless_potential > 0  # eV > 0 (吸引离子)
        repulsive_mask = dimensionless_potential <= 0  # eV <= 0 (排斥离子)
        
        J_i = torch.zeros_like(V)
        
        # 吸引性电位: J_i = J_i0 * (1 + eV/(k_B*T_i))
        if attractive_mask.any():
            J_i[attractive_mask] = J_i0[attractive_mask] * (
                1 + dimensionless_potential[attractive_mask]
            )
        
        # 排斥性电位: J_i = J_i0 * exp(-eV/(k_B*T_i))
        if repulsive_mask.any():
            J_i[repulsive_mask] = J_i0[repulsive_mask] * torch.exp(
                -dimensionless_potential[repulsive_mask]
            )
        
        return J_i
    
    def photoelectron_current(self, V: torch.Tensor, S_flux: torch.Tensor, 
                            alpha_sun: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """
        计算光电子电流密度 J_ph
        
        受正电位抑制的饱和模型:
        - V <= 0: J_ph = J_ph0
        - V > 0: J_ph = J_ph0 * exp(-V/V_ph)
        
        Args:
            V: 表面电位 (V)
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳光入射角 (度)
            material_id: 材料标识 (0: Kapton, 1: Aluminum)
            
        Returns:
            J_ph: 光电子电流密度 (A/m^2)
        """
        # 计算有效光通量 (考虑入射角)
        alpha_rad = torch.deg2rad(alpha_sun)
        effective_flux = S_flux * torch.cos(alpha_rad)
        effective_flux = torch.clamp(effective_flux, min=0.0)  # 确保非负
        
        # 根据材料类型获取光电子电流系数
        J_ph0 = torch.zeros_like(V)
        
        # Kapton材料 (material_id = 0)
        # kapton_mask = (material_id == 0)
        # if kapton_mask.any():
        #     coeff = self.materials['kapton']['J_ph0_coefficient']
        #     J_ph0[kapton_mask] = coeff * effective_flux[kapton_mask]
        
        # # Aluminum材料 (material_id = 1)
        # aluminum_mask = (material_id == 1)
        # if aluminum_mask.any():
        #     coeff = self.materials['aluminum']['J_ph0_coefficient']
        #     J_ph0[aluminum_mask] = coeff * effective_flux[aluminum_mask]
        
        # 计算光电子特征电位 V_ph = k_B*T_ph/e
        V_ph = self.k_B_eV * self.T_ph / self.e
        
        # 根据电位符号选择公式
        negative_mask = V <= 0  # V <= 0
        positive_mask = V > 0   # V > 0
        
        J_ph = torch.zeros_like(V)
        
        # V <= 0: J_ph = J_ph0
        if negative_mask.any():
            J_ph[negative_mask] = J_ph0[negative_mask]
        
        # V > 0: J_ph = J_ph0 * exp(-V/V_ph)
        if positive_mask.any():
            J_ph[positive_mask] = J_ph0[positive_mask] * torch.exp(
                -V[positive_mask] / V_ph
            )
        
        return J_ph
    
    def net_current_density(self, V: torch.Tensor, J_emission: torch.Tensor,
                          n_e: torch.Tensor, T_e: torch.Tensor,
                          n_i: torch.Tensor, T_i: torch.Tensor,
                          S_flux: torch.Tensor, alpha_sun: torch.Tensor,
                          material_id: torch.Tensor) -> torch.Tensor:
        """
        计算净电流密度 J_net
        
        J_net = J_e - J_i - J_ph - J_emission
        
        Args:
            V: 表面电位 (V)
            J_emission: 电子发射电流密度 (A/m^2) - 来自子网络
            n_e: 电子密度 (m^-3)
            T_e: 电子温度 (eV)
            n_i: 离子密度 (m^-3)
            T_i: 离子温度 (eV)
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳光入射角 (度)
            material_id: 材料标识
            
        Returns:
            J_net: 净电流密度 (A/m^2)
        """
        # 计算各个电流分量
        J_e = self.electron_current(V, n_e, T_e)
        J_i = self.ion_current(V, n_i, T_i)
        J_ph = self.photoelectron_current(V, S_flux, alpha_sun, material_id)
        
        # 计算净电流密度
        J_net = J_e - J_i - J_ph - J_emission
        
        return J_net
    
    def charging_ode(self, V: torch.Tensor, J_net: torch.Tensor) -> torch.Tensor:
        """
        航天器表面充电控制方程 (ODE)
        
        C * dV/dt = J_net * A
        
        Args:
            V: 表面电位 (V)
            J_net: 净电流密度 (A/m^2)
            
        Returns:
            dV_dt: 电位时间导数 (V/s)
        """
        dV_dt = (J_net * self.A) / self.C
        return dV_dt
    
    def physics_residual(self, V: torch.Tensor, dV_dt: torch.Tensor,
                        J_emission: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算物理残差
        
        残差 = C * dV/dt - J_net * A
        
        Args:
            V: 预测的表面电位 (V)
            dV_dt: 预测的电位时间导数 (V/s)
            J_emission: 子网络预测的发射电流密度 (A/m^2)
            inputs: 输入参数 [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
            
        Returns:
            residual: 物理残差
        """
        # 解析输入参数
        t = inputs[:, 0:1]
        n_e = inputs[:, 1:2]
        T_e = inputs[:, 2:3]
        n_i = inputs[:, 3:4]
        T_i = inputs[:, 4:5]
        S_flux = inputs[:, 5:6]
        alpha_sun = inputs[:, 6:7]
        material_id = inputs[:, 7:8]
        
        # 计算净电流密度
        J_net = self.net_current_density(
            V, J_emission, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id
        )
        
        # 计算物理残差
        residual = float(self.C) * dV_dt - J_net * self.A
        
        return residual
    
    def get_current_components(self, V: torch.Tensor, J_emission: torch.Tensor,
                             inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取所有电流分量，用于分析和可视化
        
        Args:
            V: 表面电位 (V)
            J_emission: 发射电流密度 (A/m^2)
            inputs: 输入参数
            
        Returns:
            current_components: 包含所有电流分量的字典
        """
        # 解析输入参数
        n_e = inputs[:, 1:2]
        T_e = inputs[:, 2:3]
        n_i = inputs[:, 3:4]
        T_i = inputs[:, 4:5]
        S_flux = inputs[:, 5:6]
        alpha_sun = inputs[:, 6:7]
        material_id = inputs[:, 7:8]
        
        # 计算各个电流分量
        J_e = self.electron_current(V, n_e, T_e)
        J_i = self.ion_current(V, n_i, T_i)
        J_ph = self.photoelectron_current(V, S_flux, alpha_sun, material_id)
        J_net = J_e - J_i - J_ph - J_emission
        
        return {
            'J_electron': J_e,
            'J_ion': J_i,
            'J_photoelectron': J_ph,
            'J_emission': J_emission,
            'J_net': J_net
        }

# 辅助函数
def create_physics_model(config: Dict[str, Any]) -> PhysicsModel:
    """
    创建物理模型实例
    
    Args:
        config: 配置字典
        
    Returns:
        physics_model: 物理模型实例
    """
    return PhysicsModel(config)

def validate_physics_constants(config: Dict[str, Any]) -> bool:
    """
    验证物理常数的合理性
    
    Args:
        config: 配置字典
        
    Returns:
        is_valid: 是否有效
    """
    physics_config = config['physics']
    
    # 检查基本物理常数
    required_constants = ['e', 'k_B', 'k_B_eV', 'm_e', 'm_i']
    for const in required_constants:
        if const not in physics_config or float(physics_config[const]) <= 0:
            return False
    
    # 检查航天器参数
    spacecraft_config = physics_config['spacecraft']
    required_params = ['area', 'capacitance', 'T_ph']
    for param in required_params:
        if param not in spacecraft_config or float(spacecraft_config[param]) <= 0:
            return False
    
    return True