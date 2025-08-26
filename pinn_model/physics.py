"""物理方程模块

实现航天器表面充电的所有已知物理方程和电流模型。
包括：
1. 入射等离子体电流 (J_e, J_i) - 基于轨道限制理论
2. 光电子电流 (J_ph) - 受正电位抑制的饱和模型
3. 净电流密度计算
4. 控制方程 (ODE)

Author: PINN Engineering Team
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union


class PhysicsConstants:
    """物理常数类"""
    
    def __init__(self, config: Dict):
        """初始化物理常数
        
        Args:
            config: 配置字典，包含物理常数
        """
        self.k_B = config['physics']['k_B']  # 玻尔兹曼常数 (J/K)
        self.e = config['physics']['e']      # 元电荷 (C)
        self.m_e = config['physics']['m_e']  # 电子质量 (kg)
        self.m_i = config['physics']['m_i']  # 质子质量 (kg)
        self.C = config['physics']['C']      # 等效电容 (F)
        self.A = config['physics']['A']      # 表面积 (m^2)
        self.T_ph = config['physics']['T_ph'] # 光电子特征温度 (eV)
        
        # 材料相关参数
        self.materials = config['physics']['materials']
        
        # 转换为eV单位的玻尔兹曼常数
        self.k_B_eV = self.k_B / self.e  # eV/K


class PlasmaCurrents:
    """等离子体电流计算类
    
    基于轨道限制理论计算入射电子和离子电流。
    """
    
    def __init__(self, constants: PhysicsConstants):
        self.constants = constants
    
    def thermal_current_density(self, n_s: torch.Tensor, T_s: torch.Tensor, 
                              m_s: float) -> torch.Tensor:
        """计算无电位表面的随机热电流密度 J_s0
        
        Args:
            n_s: 粒子密度 (m^-3)
            T_s: 粒子温度 (eV)
            m_s: 粒子质量 (kg)
            
        Returns:
            J_s0: 热电流密度 (A/m^2)
        """
        # 转换温度到开尔文
        T_s_K = T_s / self.constants.k_B_eV
        
        # J_s0 = n_s * q_s * sqrt(k_B * T_s / (2 * pi * m_s))
        # 对于电子: q_s = -e, 对于离子: q_s = +e
        thermal_velocity = torch.sqrt(self.constants.k_B * T_s_K / (2 * np.pi * m_s))
        J_s0 = n_s * self.constants.e * thermal_velocity
        
        return J_s0
    
    def electron_current(self, V: torch.Tensor, n_e: torch.Tensor, 
                        T_e: torch.Tensor) -> torch.Tensor:
        """计算电子电流密度 J_e
        
        Args:
            V: 表面电位 (V)
            n_e: 电子密度 (m^-3)
            T_e: 电子温度 (eV)
            
        Returns:
            J_e: 电子电流密度 (A/m^2)
        """
        J_e0 = self.thermal_current_density(n_e, T_e, self.constants.m_e)
        
        # 对于电子，q_s = -e，所以 q_s * V = -e * V
        q_V_over_kT = -self.constants.e * V / (self.constants.k_B_eV * T_e)
        
        # 吸引性电位 (q_s * V < 0, 即 V > 0): J_s = J_s0 * (1 - q_s*V/(k_B*T_s))
        # 排斥性电位 (q_s * V > 0, 即 V < 0): J_s = J_s0 * exp(q_s*V/(k_B*T_s))
        
        attractive_mask = V > 0  # 对电子来说，正电位是吸引性的
        repulsive_mask = V <= 0
        
        J_e = torch.zeros_like(V)
        
        # 吸引性电位：J_e = J_e0 * (1 + e*V/(k_B*T_e))
        if torch.any(attractive_mask):
            J_e[attractive_mask] = J_e0[attractive_mask] * (1 - q_V_over_kT[attractive_mask])
        
        # 排斥性电位：J_e = J_e0 * exp(-e*V/(k_B*T_e))
        if torch.any(repulsive_mask):
            J_e[repulsive_mask] = J_e0[repulsive_mask] * torch.exp(q_V_over_kT[repulsive_mask])
        
        return J_e
    
    def ion_current(self, V: torch.Tensor, n_i: torch.Tensor, 
                   T_i: torch.Tensor) -> torch.Tensor:
        """计算离子电流密度 J_i
        
        Args:
            V: 表面电位 (V)
            n_i: 离子密度 (m^-3)
            T_i: 离子温度 (eV)
            
        Returns:
            J_i: 离子电流密度 (A/m^2)
        """
        J_i0 = self.thermal_current_density(n_i, T_i, self.constants.m_i)
        
        # 对于离子，q_s = +e，所以 q_s * V = +e * V
        q_V_over_kT = self.constants.e * V / (self.constants.k_B_eV * T_i)
        
        attractive_mask = V < 0  # 对离子来说，负电位是吸引性的
        repulsive_mask = V >= 0
        
        J_i = torch.zeros_like(V)
        
        # 吸引性电位：J_i = J_i0 * (1 - e*V/(k_B*T_i))
        if torch.any(attractive_mask):
            J_i[attractive_mask] = J_i0[attractive_mask] * (1 - q_V_over_kT[attractive_mask])
        
        # 排斥性电位：J_i = J_i0 * exp(e*V/(k_B*T_i))
        if torch.any(repulsive_mask):
            J_i[repulsive_mask] = J_i0[repulsive_mask] * torch.exp(q_V_over_kT[repulsive_mask])
        
        return J_i


class PhotoelectronCurrent:
    """光电子电流计算类
    
    实现受正电位抑制的饱和光电子电流模型。
    """
    
    def __init__(self, constants: PhysicsConstants):
        self.constants = constants
    
    def calculate_J_ph0(self, S_flux: torch.Tensor, alpha_sun: torch.Tensor, 
                       material_id: torch.Tensor) -> torch.Tensor:
        """计算饱和光电子电流密度 J_ph0
        
        Args:
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳入射角 (degrees)
            material_id: 材料标识 (0: Kapton, 1: Aluminum)
            
        Returns:
            J_ph0: 饱和光电子电流密度 (A/m^2)
        """
        # 转换角度到弧度
        alpha_rad = alpha_sun * np.pi / 180.0
        
        # 有效太阳通量（考虑入射角）
        effective_flux = S_flux * torch.cos(alpha_rad)
        effective_flux = torch.clamp(effective_flux, min=0.0)  # 确保非负
        
        # 根据材料类型选择光电子系数
        J_ph0 = torch.zeros_like(S_flux)
        
        # Kapton (material_id = 0)
        kapton_mask = (material_id == 0)
        if torch.any(kapton_mask):
            coeff = self.constants.materials['kapton']['J_ph0_coefficient']
            J_ph0[kapton_mask] = coeff * effective_flux[kapton_mask]
        
        # Aluminum (material_id = 1)
        aluminum_mask = (material_id == 1)
        if torch.any(aluminum_mask):
            coeff = self.constants.materials['aluminum']['J_ph0_coefficient']
            J_ph0[aluminum_mask] = coeff * effective_flux[aluminum_mask]
        
        return J_ph0
    
    def photoelectron_current(self, V: torch.Tensor, S_flux: torch.Tensor, 
                            alpha_sun: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """计算光电子电流密度 J_ph
        
        Args:
            V: 表面电位 (V)
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳入射角 (degrees)
            material_id: 材料标识
            
        Returns:
            J_ph: 光电子电流密度 (A/m^2)
        """
        J_ph0 = self.calculate_J_ph0(S_flux, alpha_sun, material_id)
        
        # 等效光电子温度电压
        V_ph = self.constants.k_B_eV * self.constants.T_ph  # eV转换为V
        
        # 当 V <= 0: J_ph = J_ph0
        # 当 V > 0: J_ph = J_ph0 * exp(-V/V_ph)
        
        negative_mask = V <= 0
        positive_mask = V > 0
        
        J_ph = torch.zeros_like(V)
        
        if torch.any(negative_mask):
            J_ph[negative_mask] = J_ph0[negative_mask]
        
        if torch.any(positive_mask):
            J_ph[positive_mask] = J_ph0[positive_mask] * torch.exp(-V[positive_mask] / V_ph)
        
        return J_ph


class SpacecraftChargingPhysics:
    """航天器表面充电物理模型主类
    
    整合所有物理模型，计算净电流密度和控制方程。
    """
    
    def __init__(self, config: Dict):
        """初始化物理模型
        
        Args:
            config: 配置字典
        """
        self.constants = PhysicsConstants(config)
        self.plasma_currents = PlasmaCurrents(self.constants)
        self.photoelectron_current = PhotoelectronCurrent(self.constants)
    
    def net_current_density(self, V: torch.Tensor, n_e: torch.Tensor, T_e: torch.Tensor,
                          n_i: torch.Tensor, T_i: torch.Tensor, S_flux: torch.Tensor,
                          alpha_sun: torch.Tensor, material_id: torch.Tensor,
                          J_emission: torch.Tensor) -> torch.Tensor:
        """计算净电流密度 J_net
        
        Args:
            V: 表面电位 (V)
            n_e: 电子密度 (m^-3)
            T_e: 电子温度 (eV)
            n_i: 离子密度 (m^-3)
            T_i: 离子温度 (eV)
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳入射角 (degrees)
            material_id: 材料标识
            J_emission: 总电子发射电流密度 (A/m^2) - 来自神经网络
            
        Returns:
            J_net: 净电流密度 (A/m^2)
        """
        # 计算各项电流
        J_e = self.plasma_currents.electron_current(V, n_e, T_e)
        J_i = self.plasma_currents.ion_current(V, n_i, T_i)
        J_ph = self.photoelectron_current.photoelectron_current(V, S_flux, alpha_sun, material_id)
        
        # 净电流密度：J_net = J_e - J_i - J_ph - J_emission
        # 注意符号约定：流入表面为正，流出表面为负
        J_net = J_e - J_i - J_ph - J_emission
        
        return J_net
    
    def charging_ode(self, V: torch.Tensor, dV_dt: torch.Tensor, 
                    n_e: torch.Tensor, T_e: torch.Tensor, n_i: torch.Tensor, T_i: torch.Tensor,
                    S_flux: torch.Tensor, alpha_sun: torch.Tensor, material_id: torch.Tensor,
                    J_emission: torch.Tensor) -> torch.Tensor:
        """航天器表面充电控制方程（ODE）
        
        控制方程：C * dV/dt = J_net * A
        
        Args:
            V: 表面电位 (V)
            dV_dt: 电位时间导数 (V/s)
            其他参数同 net_current_density
            
        Returns:
            residual: 物理残差
        """
        J_net = self.net_current_density(V, n_e, T_e, n_i, T_i, S_flux, 
                                       alpha_sun, material_id, J_emission)
        
        # 控制方程：C * dV/dt = J_net * A
        # 残差：C * dV/dt - J_net * A = 0
        residual = self.constants.C * dV_dt - J_net * self.constants.A
        
        return residual
    
    def get_physics_info(self) -> Dict:
        """获取物理模型信息
        
        Returns:
            info: 包含物理常数和模型信息的字典
        """
        return {
            'constants': {
                'k_B': self.constants.k_B,
                'e': self.constants.e,
                'm_e': self.constants.m_e,
                'm_i': self.constants.m_i,
                'C': self.constants.C,
                'A': self.constants.A,
                'T_ph': self.constants.T_ph
            },
            'materials': self.constants.materials,
            'equations': [
                'Electron current: J_e (Orbital Motion Limited Theory)',
                'Ion current: J_i (Orbital Motion Limited Theory)',
                'Photoelectron current: J_ph (Suppressed saturation model)',
                'Net current: J_net = J_e - J_i - J_ph - J_emission',
                'Charging ODE: C * dV/dt = J_net * A'
            ]
        }


def create_physics_model(config: Dict) -> SpacecraftChargingPhysics:
    """创建物理模型实例的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        physics_model: 物理模型实例
    """
    return SpacecraftChargingPhysics(config)