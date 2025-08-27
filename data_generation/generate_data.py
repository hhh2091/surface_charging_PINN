# 数据生成脚本
# 模拟生成高保真训练/测试数据

import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class SpacecraftChargingSimulator:
    """
    航天器表面充电仿真器
    
    使用已知的物理模型生成高保真数据，用于训练PINN模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化仿真器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.physics_config = config['physics']
        self.data_config = config['data_generation']
        
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
        
        logging.info("Spacecraft charging simulator initialized")
    
    def thermal_current_density(self, n: float, T: float, mass: float, charge: float) -> float:
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
        T_kelvin = float(T) / self.k_B_eV
        
        # J_s0 = n * |q_s| * sqrt(k_B * T_s / (2 * pi * m_s))
        thermal_velocity = np.sqrt(self.k_B * T_kelvin / (2 * np.pi * mass))
        J_s0 = float(n) * abs(float(charge)) * thermal_velocity
        
        return float(J_s0)
    
    def electron_current(self, V: float, n_e: float, T_e: float) -> float:
        """
        计算电子电流密度 J_e
        
        Args:
            V: 表面电位 (V)
            n_e: 电子密度 (m^-3)
            T_e: 电子温度 (eV)
            
        Returns:
            J_e: 电子电流密度 (A/m^2)
        """
        # 确保输入为浮点数
        V, n_e, T_e = float(V), float(n_e), float(T_e)
        
        # 计算热电流密度
        J_e0 = self.thermal_current_density(n_e, T_e, self.m_e, -self.e)
        
        # 计算无量纲电位 eV/(k_B*T_e)
        dimensionless_potential = self.e * V / (self.k_B_eV * T_e)
        
        # 根据电位符号选择公式
        if dimensionless_potential < 0:  # 吸引电子
            J_e = J_e0 * (1 - dimensionless_potential)
        else:  # 排斥电子
            J_e = J_e0 * np.exp(dimensionless_potential)
        
        return float(J_e)
    
    def ion_current(self, V: float, n_i: float, T_i: float) -> float:
        """
        计算离子电流密度 J_i
        
        Args:
            V: 表面电位 (V)
            n_i: 离子密度 (m^-3)
            T_i: 离子温度 (eV)
            
        Returns:
            J_i: 离子电流密度 (A/m^2)
        """
        # 确保输入为浮点数
        V, n_i, T_i = float(V), float(n_i), float(T_i)
        
        # 计算热电流密度
        J_i0 = self.thermal_current_density(n_i, T_i, self.m_i, self.e)
        
        # 计算无量纲电位 eV/(k_B*T_i)
        dimensionless_potential = self.e * V / (self.k_B_eV * T_i)
        
        # 根据电位符号选择公式
        if dimensionless_potential > 0:  # 吸引离子
            J_i = J_i0 * (1 + dimensionless_potential)
        else:  # 排斥离子
            J_i = J_i0 * np.exp(-dimensionless_potential)
        
        return float(J_i)
    
    def photoelectron_current(self, V: float, S_flux: float, alpha_sun: float, material_id: int) -> float:
        """
        计算光电子电流密度 J_ph
        
        Args:
            V: 表面电位 (V)
            S_flux: 太阳辐射通量 (W/m^2)
            alpha_sun: 太阳光入射角 (度)
            material_id: 材料标识 (0: Kapton, 1: Aluminum)
            
        Returns:
            J_ph: 光电子电流密度 (A/m^2)
        """
        # 确保输入为浮点数
        V, S_flux, alpha_sun = float(V), float(S_flux), float(alpha_sun)
        material_id = int(material_id)
        
        # 计算有效光通量 (考虑入射角)
        alpha_rad = np.deg2rad(alpha_sun)
        effective_flux = S_flux * np.cos(alpha_rad)
        effective_flux = max(0.0, effective_flux)  # 确保非负
        
        # 根据材料类型获取光电子电流系数
        if material_id == 0:  # Kapton
            coeff = self.materials['kapton']['J_ph0_coefficient']
        elif material_id == 1:  # Aluminum
            coeff = self.materials['aluminum']['J_ph0_coefficient']
        else:
            raise ValueError(f"Unknown material_id: {material_id}")
        
        J_ph0 = float(coeff) * effective_flux
        
        # 计算光电子特征电位 V_ph = k_B*T_ph/e
        V_ph = self.k_B_eV * self.T_ph / self.e
        
        # 根据电位符号选择公式
        if V <= 0:
            J_ph = J_ph0
        else:
            J_ph = J_ph0 * np.exp(-V / V_ph)
        
        return float(J_ph)
    
    def emission_current_model(self, V: float, material_id: int) -> float:
        """
        简化的电子发射电流模型（用于生成真实数据）
        
        这个模型将被PINN的子网络学习
        
        Args:
            V: 表面电位 (V)
            material_id: 材料标识
            
        Returns:
            J_emission: 发射电流密度 (A/m^2)
        """
        # 确保输入为浮点数
        V = float(V)
        material_id = int(material_id)
        
        # 简化的二次电子发射模型
        if material_id == 0:  # Kapton
            J_sec0 = self.materials['kapton']['J_sec0']
            E_max = self.materials['kapton']['E_max']
        elif material_id == 1:  # Aluminum
            J_sec0 = self.materials['aluminum']['J_sec0']
            E_max = self.materials['aluminum']['E_max']
        else:
            raise ValueError(f"Unknown material_id: {material_id}")
        
        # 简化的发射电流模型：J_emission = J_sec0 * (1 + V/E_max) * exp(-|V|/E_max)
        if V >= 0:
            J_emission = float(J_sec0) * (1 + V / E_max) * np.exp(-V / E_max)
        else:
            J_emission = float(J_sec0) * np.exp(abs(V) / E_max)
        
        return float(J_emission)
    
    def net_current_density(self, V: float, n_e: float, T_e: float, n_i: float, T_i: float,
                          S_flux: float, alpha_sun: float, material_id: int) -> float:
        """
        计算净电流密度 J_net
        
        Args:
            V: 表面电位 (V)
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
        J_emission = self.emission_current_model(V, material_id)
        
        # 计算净电流密度
        J_net = J_e - J_i - J_ph - J_emission
        
        return float(J_net)
    
    def charging_ode(self, t: float, y: List[float], params: Dict[str, float]) -> List[float]:
        """
        航天器表面充电ODE
        
        C * dV/dt = J_net * A
        
        Args:
            t: 时间 (s)
            y: 状态变量 [V]
            params: 环境参数字典
            
        Returns:
            dydt: 状态变量导数 [dV/dt]
        """
        V = float(y[0])
        
        # 计算净电流密度
        J_net = self.net_current_density(
            V, params['n_e'], params['T_e'], params['n_i'], params['T_i'],
            params['S_flux'], params['alpha_sun'], params['material_id']
        )
        
        # 计算电位导数
        dV_dt = (J_net * self.A) / self.C
        
        return [float(dV_dt)]
    
    def solve_with_ode(self, params: Dict[str, float], t_span: Tuple[float, float],
                      t_eval: np.ndarray, V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用ODE求解器求解充电方程
        
        Args:
            params: 环境参数字典
            t_span: 时间范围 (t_start, t_end)
            t_eval: 评估时间点
            V0: 初始电位 (V)
            
        Returns:
            t: 时间数组
            V: 电位数组
        """
        # 确保所有输入都是正确的类型
        t_span = (float(t_span[0]), float(t_span[1]))
        t_eval = np.array(t_eval, dtype=float)
        V0 = float(V0)
        
        try:
            # 求解ODE
            sol = solve_ivp(
                fun=lambda t, y: self.charging_ode(t, y, params),
                t_span=t_span,
                y0=[V0],
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if sol.success:
                return sol.t.astype(float), sol.y[0].astype(float)
            else:
                logging.warning(f"ODE solver failed: {sol.message}")
                # 返回近似解
                return self.approximate_charging_solution(params, t_eval, V0)
        
        except Exception as e:
            logging.warning(f"ODE solver error: {e}")
            # 返回近似解
            return self.approximate_charging_solution(params, t_eval, V0)
    
    def approximate_charging_solution(self, params: Dict[str, float], 
                                    t_eval: np.ndarray, V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用前向欧拉法近似求解充电方程
        
        Args:
            params: 环境参数字典
            t_eval: 评估时间点
            V0: 初始电位 (V)
            
        Returns:
            t: 时间数组
            V: 电位数组
        """
        # 确保输入类型正确
        t_eval = np.array(t_eval, dtype=float)
        V0 = float(V0)
        
        t = t_eval.copy()
        V = np.zeros_like(t, dtype=float)
        V[0] = V0
        
        # 前向欧拉法
        for i in range(1, len(t)):
            dt = float(t[i] - t[i-1])
            
            # 计算当前时刻的导数
            dV_dt = self.charging_ode(float(t[i-1]), [float(V[i-1])], params)[0]
            
            # 更新电位
            V[i] = float(V[i-1]) + float(dV_dt) * dt
        
        return t, V
    
    def analytical_approximation(self, params: Dict[str, float], 
                               t_eval: np.ndarray, V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析近似解（适用于某些简化情况）
        
        Args:
            params: 环境参数字典
            t_eval: 评估时间点
            V0: 初始电位 (V)
            
        Returns:
            t: 时间数组
            V: 电位数组
        """
        # 确保输入类型正确
        t_eval = np.array(t_eval, dtype=float)
        V0 = float(V0)
        
        # 计算平衡电位（简化假设）
        V_eq = 0.0  # 简化假设平衡电位为0
        
        # 计算时间常数（简化估计）
        tau = float(self.C) / (float(self.A) * 1e-6)  # 简化时间常数估计
        
        # 指数衰减到平衡态
        V_solution = np.zeros_like(t_eval, dtype=float)
        for i, t in enumerate(t_eval):
            t = float(t)
            V_solution[i] = float(V_eq) + (float(V0) - float(V_eq)) * np.exp(-t / float(tau))
        
        return t_eval, V_solution
    
    def generate_scenario_data(self, scenario_params: Dict[str, float], 
                             method: str = 'ode') -> pd.DataFrame:
        """
        生成单个场景的时间序列数据
        
        Args:
            scenario_params: 场景参数字典
            method: 求解方法 ('ode', 'euler', 'analytical')
            
        Returns:
            df: 包含时间序列数据的DataFrame
        """
        # 生成时间点
        t_start = self.data_config['t_start']
        t_end = self.data_config['t_end']
        n_points = self.data_config['n_time_points']
        
        t_eval = np.linspace(t_start, t_end, n_points)
        
        # 选择求解方法
        if method == 'ode':
            t, V = self.solve_with_ode(scenario_params, (t_start, t_end), t_eval)
        elif method == 'euler':
            t, V = self.approximate_charging_solution(scenario_params, t_eval)
        elif method == 'analytical':
            t, V = self.analytical_approximation(scenario_params, t_eval)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 创建DataFrame
        data = {
            't': t,
            'n_e': [scenario_params['n_e']] * len(t),
            'T_e': [scenario_params['T_e']] * len(t),
            'n_i': [scenario_params['n_i']] * len(t),
            'T_i': [scenario_params['T_i']] * len(t),
            'S_flux': [scenario_params['S_flux']] * len(t),
            'alpha_sun': [scenario_params['alpha_sun']] * len(t),
            'material_id': [scenario_params['material_id']] * len(t),
            'V_ground_truth': V
        }
        
        df = pd.DataFrame(data)
        return df
    
    def generate_random_scenarios(self, num_scenarios: int) -> List[Dict[str, float]]:
        """
        生成随机场景参数
        
        Args:
            num_scenarios: 场景数量
            
        Returns:
            scenarios: 场景参数列表
        """
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = {
                'n_e': np.random.uniform(float(self.data_config['n_e_range'][0]), float(self.data_config['n_e_range'][1])),
                'T_e': np.random.uniform(float(self.data_config['T_e_range'][0]), float(self.data_config['T_e_range'][1])),
                'n_i': np.random.uniform(float(self.data_config['n_i_range'][0]), float(self.data_config['n_i_range'][1])),
                'T_i': np.random.uniform(float(self.data_config['T_i_range'][0]), float(self.data_config['T_i_range'][1])),
                'S_flux': np.random.uniform(float(self.data_config['S_flux_range'][0]), float(self.data_config['S_flux_range'][1])),
                'alpha_sun': np.random.uniform(float(self.data_config['alpha_sun_range'][0]), float(self.data_config['alpha_sun_range'][1])),
                'material_id': np.random.choice(self.data_config['materials'])
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_special_test_scenarios(self) -> List[Dict[str, float]]:
        """
        生成特殊测试场景（用于物理核查）
        
        Returns:
            scenarios: 特殊测试场景列表
        """
        scenarios = []
        
        # 阴影区场景（无太阳辐射）
        shadow_scenario = {
            'n_e': 1e6,
            'T_e': 1.0,
            'n_i': 1e6,
            'T_i': 0.5,
            'S_flux': 0.0,  # 无太阳辐射
            'alpha_sun': 0.0,
            'material_id': 0
        }
        scenarios.append(shadow_scenario)
        
        # 光照区场景（标准太阳常数）
        sunlit_scenario = {
            'n_e': 1e6,
            'T_e': 2.0,
            'n_i': 5e5,
            'T_i': 1.0,
            'S_flux': 1360.0,  # 标准太阳常数
            'alpha_sun': 0.0,   # 垂直入射
            'material_id': 1
        }
        scenarios.append(sunlit_scenario)
        
        # 高能等离子体环境
        high_energy_scenario = {
            'n_e': 1e7,
            'T_e': 5.0,
            'n_i': 1e7,
            'T_i': 3.0,
            'S_flux': 500.0,
            'alpha_sun': 45.0,
            'material_id': 0
        }
        scenarios.append(scenarios)
        
        return scenarios

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
            logging.FileHandler('data_generation.log')
        ]
    )

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Generate spacecraft charging data')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output_dir', type=str, default='data/',
                       help='Output directory for generated data')
    parser.add_argument('--train_scenarios', type=int, default=None,
                       help='Number of training scenarios')
    parser.add_argument('--test_scenarios', type=int, default=None,
                       help='Number of test scenarios')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['ode', 'euler', 'analytical', 'auto'],
                       help='Solving method')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置中的参数（如果命令行提供）
    if args.train_scenarios is not None:
        config['data_generation']['train_scenarios'] = args.train_scenarios
    if args.test_scenarios is not None:
        config['data_generation']['test_scenarios'] = args.test_scenarios
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化仿真器
    simulator = SpacecraftChargingSimulator(config)
    
    # 生成训练数据
    logging.info("Generating training data...")
    train_scenarios = simulator.generate_random_scenarios(
        config['data_generation']['train_scenarios']
    )
    args.method = 'ode'
    train_data_list = []
    for i, scenario in enumerate(train_scenarios):
        try:
            # 选择求解方法
            if args.method == 'auto':
                # 自动选择最适合的方法
                method = 'ode' if i % 3 == 0 else ('euler' if i % 3 == 1 else 'analytical')
            else:
                method = args.method
            
            df = simulator.generate_scenario_data(scenario, method=method)
            train_data_list.append(df)
            
            if (i + 1) % 100 == 0:
                logging.info(f"Generated {i + 1}/{len(train_scenarios)} training scenarios")
        
        except Exception as e:
            logging.error(f"Error generating training scenario {i}: {e}")
            continue
    
    # 合并训练数据
    if train_data_list:
        train_data = pd.concat(train_data_list, ignore_index=True)
        train_path = output_dir / 'train_data.csv'
        train_data.to_csv(train_path, index=False)
        logging.info(f"Training data saved to {train_path} ({len(train_data)} samples)")
    else:
        logging.warning("No training data generated")
        # 保存空数据集
        empty_df = pd.DataFrame(columns=[
            't', 'n_e', 'T_e', 'n_i', 'T_i', 'S_flux', 'alpha_sun', 'material_id', 'V_ground_truth'
        ])
        train_path = output_dir / 'train_data.csv'
        empty_df.to_csv(train_path, index=False)
    
    # 生成测试数据
    logging.info("Generating test data...")
    test_scenarios = simulator.generate_random_scenarios(
        config['data_generation']['test_scenarios']
    )
    
    test_data_list = []
    for i, scenario in enumerate(test_scenarios):
        try:
            # 选择求解方法
            if args.method == 'auto':
                method = 'ode' if i % 2 == 0 else 'euler'
            else:
                method = args.method
            
            df = simulator.generate_scenario_data(scenario, method=method)
            test_data_list.append(df)
            
            if (i + 1) % 50 == 0:
                logging.info(f"Generated {i + 1}/{len(test_scenarios)} test scenarios")
        
        except Exception as e:
            logging.error(f"Error generating test scenario {i}: {e}")
            continue
    
    # 合并测试数据
    if test_data_list:
        test_data = pd.concat(test_data_list, ignore_index=True)
        test_path = output_dir / 'test_data.csv'
        test_data.to_csv(test_path, index=False)
        logging.info(f"Test data saved to {test_path} ({len(test_data)} samples)")
    else:
        logging.warning("No test data generated")
        # 保存空数据集
        empty_df = pd.DataFrame(columns=[
            't', 'n_e', 'T_e', 'n_i', 'T_i', 'S_flux', 'alpha_sun', 'material_id', 'V_ground_truth'
        ])
        test_path = output_dir / 'test_data.csv'
        empty_df.to_csv(test_path, index=False)
    
    # 生成特殊测试场景
    logging.info("Generating special test scenarios...")
    special_scenarios = simulator.generate_special_test_scenarios()
    
    special_data_list = []
    for i, scenario in enumerate(special_scenarios):
        try:
            df = simulator.generate_scenario_data(scenario, method='ode')
            special_data_list.append(df)
        except Exception as e:
            logging.error(f"Error generating special scenario {i}: {e}")
            continue
    
    # 合并特殊测试数据
    if special_data_list:
        special_data = pd.concat(special_data_list, ignore_index=True)
        special_path = output_dir / 'special_test_data.csv'
        special_data.to_csv(special_path, index=False)
        logging.info(f"Special test data saved to {special_path} ({len(special_data)} samples)")
    
    logging.info("Data generation completed successfully!")

if __name__ == '__main__':
    main()