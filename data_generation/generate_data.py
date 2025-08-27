"""数据生成脚本

生成航天器表面充电PINN训练所需的模拟数据。
该脚本创建高保真的合成数据集，包含各种环境条件下的表面电位数据。

生成的数据包含以下列：
[t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id, V_ground_truth]

Author: PINN Engineering Team
"""

import numpy as np
import pandas as pd
import yaml
import os
import argparse
from typing import Dict, Tuple, List
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm


class SyntheticDataGenerator:
    """合成数据生成器
    
    使用简化的物理模型生成训练数据。
    """
    
    def __init__(self, config: Dict, solver_method: str = 'auto'):
        """初始化数据生成器
        
        Args:
            config: 配置字典
            solver_method: 求解方法 ('approximate', 'ode', 'analytical', 'auto')
        """
        self.config = config
        self.data_config = config['data_generation']
        self.physics_config = config['physics']
        self.solver_method = solver_method
        
        # 物理常数
        self.k_B = self.physics_config['k_B']
        self.e = self.physics_config['e']
        self.m_e = self.physics_config['m_e']
        self.m_i = self.physics_config['m_i']
        self.C = float(self.physics_config['C'])
        self.A = float(self.physics_config['A'])
        self.T_ph = float(self.physics_config['T_ph'])
        
        # eV单位的玻尔兹曼常数
        self.k_B_eV = self.k_B / self.e
        
        print(f"Using solver method: {solver_method}")
    
    def thermal_current_density(self, n_s: float, T_s: float, m_s: float) -> float:
        """计算热电流密度
        
        Args:
            n_s: 粒子密度 (m^-3)
            T_s: 粒子温度 (eV)
            m_s: 粒子质量 (kg)
            
        Returns:
            J_s0: 热电流密度 (A/m^2)
        """
        n_s = float(n_s)
        T_s = float(T_s)
        m_s = float(m_s)
        T_s_K = float(T_s / self.k_B_eV)
        thermal_velocity = float(np.sqrt(self.k_B * T_s_K / (2 * np.pi * m_s)))
        return float(n_s * self.e * thermal_velocity)
    
    def electron_current(self, V: float, n_e: float, T_e: float) -> float:
        """计算电子电流密度"""
        J_e0 = self.thermal_current_density(n_e, T_e, self.m_e)
        q_V_over_kT = float(-self.e * V / (self.k_B_eV * T_e))
        
        if V > 0:  # 吸引性电位
            return float(J_e0 * (1 - q_V_over_kT))
        else:  # 排斥性电位
            return float(J_e0 * np.exp(q_V_over_kT))
    
    def ion_current(self, V: float, n_i: float, T_i: float) -> float:
        """计算离子电流密度"""
        J_i0 = self.thermal_current_density(n_i, T_i, self.m_i)
        q_V_over_kT = float(self.e * V / (self.k_B_eV * T_i))
        
        if V < 0:  # 吸引性电位
            return float(J_i0 * (1 - q_V_over_kT))
        else:  # 排斥性电位
            return float(J_i0 * np.exp(q_V_over_kT))
    
    def photoelectron_current(self, V: float, S_flux: float, alpha_sun: float, 
                            material_id: int) -> float:
        """计算光电子电流密度"""
        # 获取材料参数
        if material_id == 0:  # Kapton
            coeff = float(self.physics_config['materials']['kapton']['J_ph0_coefficient'])
        else:  # Aluminum
            coeff = float(self.physics_config['materials']['aluminum']['J_ph0_coefficient'])
        
        # 有效太阳通量
        alpha_rad = float(np.deg2rad(float(alpha_sun)))
        effective_flux = float(S_flux) * float(np.cos(alpha_rad))
        effective_flux = max(0.0, effective_flux)
        
        J_ph0 = float(coeff) * float(effective_flux)
        
        # 电位抑制
        V_ph = float(self.k_B_eV * self.T_ph)
        if float(V) <= 0:
            return float(J_ph0)
        else:
            return float(J_ph0) * float(np.exp(-float(V) / V_ph))
    
    def emission_current_model(self, V: float, material_id: int) -> float:
        """简化的电子发射电流模型
        
        这是一个简化的经验模型，用于生成训练数据。
        实际的发射电流将由神经网络学习。
        
        Args:
            V: 表面电位 (V)
            material_id: 材料标识
            
        Returns:
            J_emission: 电子发射电流密度 (A/m^2)
        """
        # 简化的二次电子发射和背散射电子模型
        V = float(V)
        material_id = int(material_id)
        
        if material_id == 0:  # Kapton - 高SEY材料
            if V < -50:
                return float(1e-6 * np.exp(-V / 10))  # 强电场发射
            elif V < 0:
                return float(1e-8 * (-V) ** 0.5)  # 弱场发射
            else:
                return float(1e-9 * (1 + V / 5))  # 正电位下的热发射
        else:  # Aluminum - 低SEY材料
            if V < -100:
                return float(5e-7 * np.exp(-V / 15))
            elif V < 0:
                return float(5e-9 * (-V) ** 0.3)
            else:
                return float(1e-10 * (1 + V / 10))
    
    def net_current_density(self, V: float, params: Dict) -> float:
        """计算净电流密度
        
        Args:
            V: 表面电位 (V)
            params: 环境参数字典
            
        Returns:
            J_net: 净电流密度 (A/m^2)
        """
        # 确保电位是浮点数
        V = float(V)
        
        # 计算各分量电流
        J_e = float(self.electron_current(V, float(params['n_e']), float(params['T_e'])))
        J_i = float(self.ion_current(V, float(params['n_i']), float(params['T_i'])))
        J_ph = float(self.photoelectron_current(V, float(params['S_flux']), float(params['alpha_sun']), 
                                        int(params['material_id'])))
        J_emission = float(self.emission_current_model(V, int(params['material_id'])))
        
        # 确保返回浮点数
        return float(J_e - J_i - J_ph - J_emission)
    
    def charging_ode(self, t: float, y: List[float], params: Dict) -> List[float]:
        """充电ODE方程
        
        Args:
            t: 时间
            y: 状态变量 [V]
            params: 环境参数
            
        Returns:
            dydt: 状态导数 [dV/dt]
        """
        V = float(y[0])
        J_net = float(self.net_current_density(V, params))
        dV_dt = float(J_net * self.A / self.C)
        return [float(dV_dt)]
    
    def solve_charging_dynamics(self, params: Dict, t_span: Tuple[float, float], 
                              V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """求解充电动力学
        
        Args:
            params: 环境参数
            t_span: 时间范围 (t_start, t_end)
            V0: 初始电位
            
        Returns:
            t_eval: 时间点数组
            V_solution: 电位解数组
        """
        t_eval = np.linspace(t_span[0], t_span[1], self.data_config['n_time_points'])
        
        # 根据指定的求解方法选择求解器
        if self.solver_method == 'approximate':
            return self.approximate_charging_solution(params, t_eval, V0)
        elif self.solver_method == 'ode':
            return self.solve_with_ode(params, t_span, t_eval, V0)
        elif self.solver_method == 'analytical':
            return self.analytical_approximation(params, t_eval, V0)
        else:  # 'auto' - 尝试所有方法
            try:
                return self.approximate_charging_solution(params, t_eval, V0)
            except Exception as e:
                print(f"Approximation method failed: {e}, falling back to ODE solver")
                
                try:
                    return self.solve_with_ode(params, t_span, t_eval, V0)
                except Exception as e:
                    print(f"ODE solver also failed: {e}, using analytical approximation")
                    return self.analytical_approximation(params, t_eval, V0)
    
    def solve_with_ode(self, params: Dict, t_span: Tuple[float, float], t_eval: np.ndarray, 
                      V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """使用ODE求解器求解充电动力学
        
        Args:
            params: 环境参数
            t_span: 时间范围
            t_eval: 评估时间点
            V0: 初始电位
            
        Returns:
            t_points: 时间点数组
            V_solution: 电位解数组
        """
        # 确保所有输入都是正确的类型
        t_span_float = (float(t_span[0]), float(t_span[1]))
        t_eval_float = np.array(t_eval, dtype=float)
        V0_float = float(V0)
        
        # 求解ODE
        sol = solve_ivp(
            fun=lambda t, y: self.charging_ode(float(t), [float(y[0])], params),
            t_span=t_span_float,
            y0=[V0_float],
            t_eval=t_eval_float,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        if not sol.success:
            print(f"Warning: ODE solver failed for params {params}")
            # 返回零解作为备选
            return t_eval, np.zeros_like(t_eval)
        
        return np.array(sol.t, dtype=float), np.array(sol.y[0], dtype=float)
    
    def approximate_charging_solution(self, params: Dict, t_eval: np.ndarray, V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """使用近似方法求解充电动力学
        
        使用前向欧拉法和自适应步长来近似求解充电方程
        
        Args:
            params: 环境参数
            t_eval: 评估时间点
            V0: 初始电位
            
        Returns:
            t_eval: 时间点数组
            V_solution: 电位解数组
        """
        # 确保输入类型正确
        t_eval = np.array(t_eval, dtype=float)
        V0 = float(V0)
        
        # 初始化解数组
        V_solution = np.zeros_like(t_eval, dtype=float)
        V_solution[0] = V0
        
        # 前向欧拉法求解
        for i in range(1, len(t_eval)):
            dt = float(t_eval[i] - t_eval[i-1])  # 确保dt是浮点数
            
            # 计算当前电位下的净电流密度
            J_net = float(self.net_current_density(float(V_solution[i-1]), params))
            
            # 计算电位变化率
            dV_dt = float(J_net * self.A / self.C)
            
            # 自适应步长：如果变化太大，使用多个小步长
            max_dV = 10.0  # 最大允许的电位变化
            if abs(dV_dt * dt) > max_dV:
                n_substeps = int(abs(dV_dt * dt) / max_dV) + 1
                sub_dt = float(dt / float(n_substeps))
                V_temp = float(V_solution[i-1])
                
                for _ in range(n_substeps):
                    J_net = float(self.net_current_density(float(V_temp), params))
                    dV_dt = float(J_net * self.A / self.C)
                    V_temp = float(V_temp + dV_dt * sub_dt)
                
                V_solution[i] = float(V_temp)
            else:
                # 标准欧拉步进
                V_solution[i] = float(V_solution[i-1]) + dV_dt * dt
        
        return t_eval, V_solution
    
    def analytical_approximation(self, params: Dict, t_eval: np.ndarray, V0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """使用解析近似求解充电方程
        
        当数值方法失败时的最后备选方案
        
        Args:
            params: 环境参数
            t_eval: 评估时间点
            V0: 初始电位
            
        Returns:
            t_eval: 时间点数组
            V_solution: 电位解数组
        """
        # 初始化解数组
        V_solution = np.zeros_like(t_eval, dtype=float)
        V_solution[0] = float(V0)
        
        # 估计平衡电位
        V_eq = float(self.estimate_equilibrium_potential(params))
        
        # 估计时间常数
        tau = float(self.estimate_time_constant(params, V_eq))
        
        # 使用指数趋近模型
        for i in range(1, len(t_eval)):
            t = float(t_eval[i] - t_eval[0])  # 相对时间
            V_solution[i] = V_eq - (V_eq - V0) * np.exp(-t / tau)
        
        return t_eval, V_solution
    
    def estimate_equilibrium_potential(self, params: Dict) -> float:
        """估计平衡电位
        
        通过二分法寻找净电流为零的电位
        
        Args:
            params: 环境参数
            
        Returns:
            V_eq: 估计的平衡电位
        """
        # 初始搜索范围
        V_min, V_max = -1000.0, 1000.0
        
        # 检查边界点的电流符号
        J_min = self.net_current_density(V_min, params)
        J_max = self.net_current_density(V_max, params)
        
        # 如果边界点电流同号，返回电流较小的边界点
        if J_min * J_max > 0:
            if abs(J_min) < abs(J_max):
                return V_min
            else:
                return V_max
        
        # 二分法搜索
        tol = 1e-6
        max_iter = 50
        
        for _ in range(max_iter):
            V_mid = (V_min + V_max) / 2
            J_mid = self.net_current_density(V_mid, params)
            
            if abs(J_mid) < tol:
                return V_mid
            
            if J_mid * J_min < 0:
                V_max = V_mid
            else:
                V_min = V_mid
                J_min = J_mid
        
        return (V_min + V_max) / 2
    
    def estimate_time_constant(self, params: Dict, V_eq: float) -> float:
        """估计充电时间常数
        
        Args:
            params: 环境参数
            V_eq: 平衡电位
            
        Returns:
            tau: 估计的时间常数
        """
        # 计算平衡点附近的电流导数
        delta_V = 0.1
        J1 = self.net_current_density(V_eq - delta_V, params)
        J2 = self.net_current_density(V_eq + delta_V, params)
        
        # 计算电导率
        conductivity = abs((J2 - J1) / (2 * delta_V))
        
        # 防止除零
        if conductivity < 1e-12:
            conductivity = 1e-12
        
        # 时间常数 = 电容 / 电导率
        tau = self.C / (conductivity * self.A)
        
        # 限制时间常数在合理范围内
        tau = max(1.0, min(tau, 3600.0))  # 1秒到1小时
        
        return tau
    
    def generate_parameter_combinations(self, n_samples: int) -> List[Dict]:
        """生成参数组合
        
        Args:
            n_samples: 样本数量
            
        Returns:
            param_combinations: 参数组合列表
        """
        combinations = []
        
        for _ in range(n_samples):
            # 随机采样环境参数
            n_e_min, n_e_max = self.data_config['n_e_range']
            n_e_min = float(n_e_min)
            n_e_max = float(n_e_max)
            n_e = np.random.uniform(n_e_min, n_e_max)
            
            T_e_min, T_e_max = self.data_config['T_e_range']
            T_e = np.random.uniform(T_e_min, T_e_max)
            
            n_i_min, n_i_max = self.data_config['n_i_range']
            n_i_min = float(n_i_min)
            n_i_max = float(n_i_max)
            n_i = np.random.uniform(n_i_min, n_i_max)
            
            T_i_min, T_i_max = self.data_config['T_i_range']
            T_i = np.random.uniform(T_i_min, T_i_max)
            
            S_flux_min, S_flux_max = self.data_config['S_flux_range']
            S_flux = np.random.uniform(S_flux_min, S_flux_max)
            
            alpha_min, alpha_max = self.data_config['alpha_sun_range']
            alpha_sun = np.random.uniform(alpha_min, alpha_max)
            
            # 随机选择材料
            materials = self.data_config['materials']
            material_id = np.random.choice(materials)
            
            combinations.append({
                'n_e': n_e,
                'T_e': T_e,
                'n_i': n_i,
                'T_i': T_i,
                'S_flux': S_flux,
                'alpha_sun': alpha_sun,
                'material_id': material_id
            })

        # print(combinations)
        
        return combinations
    
    def generate_dataset(self, n_scenarios: int, output_path: str, 
                        dataset_type: str = 'train') -> pd.DataFrame:
        """生成完整数据集
        
        Args:
            n_scenarios: 场景数量
            output_path: 输出文件路径
            dataset_type: 数据集类型 ('train', 'test')
            
        Returns:
            dataset: 生成的数据集
        """
        print(f"Generating {dataset_type} dataset with {n_scenarios} scenarios...")
        
        # 生成参数组合
        param_combinations = self.generate_parameter_combinations(n_scenarios)
        
        # 存储所有数据
        all_data = []
        
        # 时间范围
        t_start = self.data_config['t_start']
        t_end = self.data_config['t_end']
        
        for i, params in enumerate(tqdm(param_combinations, desc=f"Generating {dataset_type} data")):
            try:
                # 求解充电动力学
                t_points, V_solution = self.solve_charging_dynamics(
                    params, (t_start, t_end)
                )
                
                # 为每个时间点创建数据行
                for t, V in zip(t_points, V_solution):
                    data_row = {
                        't': t,
                        'n_e': params['n_e'],
                        'T_e': params['T_e'],
                        'n_i': params['n_i'],
                        'T_i': params['T_i'],
                        'S_flux': params['S_flux'],
                        'alpha_sun': params['alpha_sun'],
                        'material_id': params['material_id'],
                        'V_ground_truth': V
                    }
                    all_data.append(data_row)
            
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
                continue
        
        # 创建DataFrame
        dataset = pd.DataFrame(all_data)
        
        # 数据质量检查
        print(f"Generated {len(dataset)} data points")
        
        # 检查数据集是否为空
        if len(dataset) > 0:
            print(f"Voltage range: [{dataset['V_ground_truth'].min():.3f}, {dataset['V_ground_truth'].max():.3f}] V")
            print(f"Material distribution: {dataset['material_id'].value_counts().to_dict()}")
            
            # 保存数据
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dataset.to_csv(output_path, index=False)
            print(f"Dataset saved to: {output_path}")
        else:
            print("Warning: No data points were generated. Check solver method and parameters.")
            # 创建一个空的数据框架并保存
            empty_columns = ['t', 'n_e', 'T_e', 'n_i', 'T_i', 'S_flux', 'alpha_sun', 'material_id', 'V_ground_truth']
            empty_df = pd.DataFrame(columns=empty_columns)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            empty_df.to_csv(output_path, index=False)
            print(f"Empty dataset saved to: {output_path}")
        
        return dataset
    
    def generate_special_scenarios(self, output_path: str) -> pd.DataFrame:
        """生成特殊测试场景
        
        包括：
        1. 进入地球阴影区场景
        2. 平静光照区场景
        3. 极端环境条件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            special_dataset: 特殊场景数据集
        """
        print("Generating special test scenarios...")
        
        all_data = []
        
        # 场景1：进入地球阴影区
        eclipse_params = {
            'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.5,
            'S_flux': 1360.0, 'alpha_sun': 0.0, 'material_id': 0
        }
        
        # 光照阶段
        t_points1, V_solution1 = self.solve_charging_dynamics(
            eclipse_params, (0, 1800)  # 前30分钟
        )
        
        # 阴影阶段
        eclipse_params['S_flux'] = 0.0
        t_points2, V_solution2 = self.solve_charging_dynamics(
            eclipse_params, (1800, 3600), V0=V_solution1[-1]
        )
        
        # 合并数据
        for t, V in zip(np.concatenate([t_points1, t_points2]), 
                       np.concatenate([V_solution1, V_solution2])):
            S_flux = 1360.0 if t < 1800 else 0.0
            all_data.append({
                't': t, 'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.5,
                'S_flux': S_flux, 'alpha_sun': 0.0, 'material_id': 0,
                'V_ground_truth': V
            })
        
        # 场景2：平静光照区
        quiet_params = {
            'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.5,
            'S_flux': 1360.0, 'alpha_sun': 0.0, 'material_id': 1
        }
        
        t_points, V_solution = self.solve_charging_dynamics(
            quiet_params, (0, 3600)
        )
        
        for t, V in zip(t_points, V_solution):
            all_data.append({
                't': t, 'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.5,
                'S_flux': 1360.0, 'alpha_sun': 0.0, 'material_id': 1,
                'V_ground_truth': V
            })
        
        # 创建DataFrame并保存
        special_dataset = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        special_dataset.to_csv(output_path, index=False)
        
        print(f"Special scenarios saved to: {output_path}")
        return special_dataset


def load_config(config_path: str) -> Dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate spacecraft charging PINN training data')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for generated data')
    parser.add_argument('--train_scenarios', type=int, default=None,
                       help='Number of training scenarios (overrides config)')
    parser.add_argument('--test_scenarios', type=int, default=None,
                       help='Number of test scenarios (overrides config)')
    parser.add_argument('--solver', type=str, choices=['approximate', 'ode', 'analytical', 'auto'],
                       default='auto', help='Solver method: approximate (Euler), ode (solve_ivp), analytical, or auto (try all)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建数据生成器
    generator = SyntheticDataGenerator(config, solver_method=args.solver)
    
    # 确定场景数量
    train_scenarios = args.train_scenarios or config['data_generation']['n_samples'] // 100
    test_scenarios = args.test_scenarios or config['data_generation']['n_test_samples'] // 100
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成训练数据
    train_path = os.path.join(args.output_dir, 'train_data.csv')
    train_dataset = generator.generate_dataset(train_scenarios, train_path, 'train')
    
    # 生成测试数据
    test_path = os.path.join(args.output_dir, 'test_data.csv')
    test_dataset = generator.generate_dataset(test_scenarios, test_path, 'test')
    
    # 生成特殊场景
    special_path = os.path.join(args.output_dir, 'special_scenarios.csv')
    special_dataset = generator.generate_special_scenarios(special_path)
    
    print("\nData generation completed successfully!")
    print(f"Training data: {len(train_dataset)} points")
    print(f"Test data: {len(test_dataset)} points")
    print(f"Special scenarios: {len(special_dataset)} points")


if __name__ == '__main__':
    main()