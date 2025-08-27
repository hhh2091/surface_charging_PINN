# 可视化脚本
# 实现PINN模型结果的可视化

import torch
import numpy as np
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.networks import create_pinn_model
from pinn_model.physics import PhysicsModel

class PINNVisualizer:
    """
    PINN模型可视化器
    
    功能：
    1. 模型预测结果可视化
    2. 物理场分析
    3. 参数敏感性分析
    4. 动态演化可视化
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str, device: torch.device):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
            model_path: 模型文件路径
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 加载模型
        self.model = create_pinn_model(config).to(device)
        self.load_model(model_path)
        
        # 物理模型
        self.physics_model = PhysicsModel(config)
        
        # 可视化配置
        self.viz_config = config.get('visualization', {})
        
        logging.info(f"PINN visualizer initialized on device: {device}")
    
    def load_model(self, model_path: str) -> None:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")
    
    def visualize_time_evolution(self, scenarios: List[Dict[str, Any]], 
                               output_path: str = 'time_evolution.png') -> None:
        """
        可视化时间演化
        
        Args:
            scenarios: 场景列表
            output_path: 输出文件路径
        """
        logging.info("Visualizing time evolution...")
        
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4*len(scenarios)))
        if len(scenarios) == 1:
            axes = [axes]
        
        for i, scenario in enumerate(scenarios):
            # 生成时间序列
            t_end = scenario.get('t_end', 7200)  # 默认2小时
            t_values = np.linspace(0, t_end, 200)
            
            # 构建输入数据
            inputs = []
            for t in t_values:
                input_row = [
                    t,
                    scenario['n_e'],
                    scenario['T_e'],
                    scenario['n_i'],
                    scenario['T_i'],
                    scenario['S_flux'],
                    scenario['alpha_sun'],
                    scenario['material_id']
                ]
                inputs.append(input_row)
            
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                predictions = self.model.predict(inputs).cpu().numpy().flatten()
            
            # 绘制结果
            ax = axes[i]
            ax.plot(t_values / 3600, predictions, linewidth=2, label='Surface Potential')
            
            # 添加特殊事件标记
            if 'events' in scenario:
                for event in scenario['events']:
                    event_time = event['time'] / 3600
                    ax.axvline(x=event_time, color='r', linestyle='--', 
                             alpha=0.7, label=event['label'])
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Surface Potential (V)')
            ax.set_title(f"Scenario {i+1}: {scenario.get('name', 'Unnamed')}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加场景信息
            info_text = (
                f"n_e = {scenario['n_e']:.1e} m⁻³\n"
                f"T_e = {scenario['T_e']:.1f} eV\n"
                f"S_flux = {scenario['S_flux']:.0f} W/m²\n"
                f"Material: {['Kapton', 'Aluminum'][scenario['material_id']]}"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Time evolution plot saved to {output_path}")
    
    def visualize_parameter_sensitivity(self, base_scenario: Dict[str, Any],
                                      param_name: str, param_range: Tuple[float, float],
                                      output_path: str = 'parameter_sensitivity.png') -> None:
        """
        可视化参数敏感性分析
        
        Args:
            base_scenario: 基础场景
            param_name: 参数名称
            param_range: 参数范围 (min, max)
            output_path: 输出文件路径
        """
        logging.info(f"Visualizing parameter sensitivity for {param_name}...")
        
        # 参数映射
        param_mapping = {
            'n_e': 1, 'T_e': 2, 'n_i': 3, 'T_i': 4,
            'S_flux': 5, 'alpha_sun': 6
        }
        
        if param_name not in param_mapping:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        param_idx = param_mapping[param_name]
        
        # 生成参数值
        if param_name in ['n_e', 'n_i']:
            param_values = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), 20)
        else:
            param_values = np.linspace(param_range[0], param_range[1], 20)
        
        # 时间点
        time_points = [0, 1800, 3600, 7200]  # 0, 0.5h, 1h, 2h
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, t in enumerate(time_points):
            potentials = []
            
            for param_val in param_values:
                # 构建输入
                input_row = [
                    t,
                    base_scenario['n_e'],
                    base_scenario['T_e'],
                    base_scenario['n_i'],
                    base_scenario['T_i'],
                    base_scenario['S_flux'],
                    base_scenario['alpha_sun'],
                    base_scenario['material_id']
                ]
                
                # 更新参数值
                input_row[param_idx] = param_val
                
                inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
                
                # 模型预测
                with torch.no_grad():
                    prediction = self.model.predict(inputs).cpu().numpy().item()
                
                potentials.append(prediction)
            
            # 绘制结果
            ax = axes[i]
            if param_name in ['n_e', 'n_i']:
                ax.semilogx(param_values, potentials, 'o-', linewidth=2, markersize=6)
            else:
                ax.plot(param_values, potentials, 'o-', linewidth=2, markersize=6)
            
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel('Surface Potential (V)')
            ax.set_title(f't = {t/3600:.1f} hours')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter Sensitivity Analysis: {param_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Parameter sensitivity plot saved to {output_path}")
    
    def visualize_phase_space(self, scenarios: List[Dict[str, Any]],
                            x_param: str, y_param: str,
                            output_path: str = 'phase_space.png') -> None:
        """
        可视化相空间图
        
        Args:
            scenarios: 场景列表
            x_param: X轴参数
            y_param: Y轴参数
            output_path: 输出文件路径
        """
        logging.info(f"Visualizing phase space: {x_param} vs {y_param}...")
        
        # 参数映射
        param_mapping = {
            'n_e': 1, 'T_e': 2, 'n_i': 3, 'T_i': 4,
            'S_flux': 5, 'alpha_sun': 6
        }
        
        x_idx = param_mapping[x_param]
        y_idx = param_mapping[y_param]
        
        # 生成网格
        x_values = []
        y_values = []
        potentials = []
        
        for scenario in scenarios:
            # 构建输入
            input_row = [
                3600,  # 固定时间为1小时
                scenario['n_e'],
                scenario['T_e'],
                scenario['n_i'],
                scenario['T_i'],
                scenario['S_flux'],
                scenario['alpha_sun'],
                scenario['material_id']
            ]
            
            inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                prediction = self.model.predict(inputs).cpu().numpy().item()
            
            x_values.append(input_row[x_idx])
            y_values.append(input_row[y_idx])
            potentials.append(prediction)
        
        # 创建散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(x_values, y_values, c=potentials, 
                           cmap='viridis', s=50, alpha=0.7)
        
        # 设置坐标轴
        if x_param in ['n_e', 'n_i']:
            ax.set_xscale('log')
        if y_param in ['n_e', 'n_i']:
            ax.set_yscale('log')
        
        ax.set_xlabel(f'{x_param}')
        ax.set_ylabel(f'{y_param}')
        ax.set_title(f'Phase Space: Surface Potential at t=1h')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Surface Potential (V)')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Phase space plot saved to {output_path}")
    
    def visualize_current_components(self, scenario: Dict[str, Any],
                                   output_path: str = 'current_components.png') -> None:
        """
        可视化电流分量
        
        Args:
            scenario: 场景参数
            output_path: 输出文件路径
        """
        logging.info("Visualizing current components...")
        
        # 生成时间序列
        t_end = scenario.get('t_end', 7200)
        t_values = np.linspace(0, t_end, 200)
        
        # 存储各电流分量
        J_e_values = []
        J_i_values = []
        J_ph_values = []
        J_emission_values = []
        J_net_values = []
        V_values = []
        
        for t in t_values:
            # 构建输入
            input_row = [
                t,
                scenario['n_e'],
                scenario['T_e'],
                scenario['n_i'],
                scenario['T_i'],
                scenario['S_flux'],
                scenario['alpha_sun'],
                scenario['material_id']
            ]
            
            inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
            
            # 模型预测电位
            with torch.no_grad():
                V = self.model.predict(inputs).cpu().numpy().item()
                V_values.append(V)
            
            # 计算各电流分量
            params = {
                'n_e': scenario['n_e'],
                'T_e': scenario['T_e'],
                'n_i': scenario['n_i'],
                'T_i': scenario['T_i'],
                'S_flux': scenario['S_flux'],
                'alpha_sun': scenario['alpha_sun'],
                'material_id': scenario['material_id'],
                'V': V
            }
            
            J_e = self.physics_model.electron_current(params)
            J_i = self.physics_model.ion_current(params)
            J_ph = self.physics_model.photoemission_current(params)
            
            # 使用模型预测发射电流
            emission_input = torch.tensor([[V, scenario['material_id']]], 
                                        dtype=torch.float32).to(self.device)
            with torch.no_grad():
                J_emission = self.model.yield_network(emission_input).cpu().numpy().item()
            
            J_net = J_e - J_i - J_ph - J_emission
            
            J_e_values.append(J_e)
            J_i_values.append(-J_i)  # 负号表示流出
            J_ph_values.append(-J_ph)
            J_emission_values.append(-J_emission)
            J_net_values.append(J_net)
        
        # 绘制结果
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 电流分量
        ax1.plot(t_values/3600, J_e_values, label='Electron Current (J_e)', linewidth=2)
        ax1.plot(t_values/3600, J_i_values, label='Ion Current (-J_i)', linewidth=2)
        ax1.plot(t_values/3600, J_ph_values, label='Photoemission (-J_ph)', linewidth=2)
        ax1.plot(t_values/3600, J_emission_values, label='Secondary Emission (-J_emission)', linewidth=2)
        ax1.plot(t_values/3600, J_net_values, label='Net Current (J_net)', linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Current Density (A/m²)')
        ax1.set_title('Current Components vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('symlog', linthresh=1e-10)
        
        # 表面电位
        ax2.plot(t_values/3600, V_values, 'r-', linewidth=2, label='Surface Potential')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Surface Potential (V)')
        ax2.set_title('Surface Potential vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Current components plot saved to {output_path}")
    
    def visualize_material_comparison(self, base_scenario: Dict[str, Any],
                                    output_path: str = 'material_comparison.png') -> None:
        """
        可视化材料对比
        
        Args:
            base_scenario: 基础场景
            output_path: 输出文件路径
        """
        logging.info("Visualizing material comparison...")
        
        materials = [0, 1]  # Kapton, Aluminum
        material_names = ['Kapton', 'Aluminum']
        colors = ['blue', 'red']
        
        # 生成时间序列
        t_end = base_scenario.get('t_end', 7200)
        t_values = np.linspace(0, t_end, 200)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for mat_id, mat_name, color in zip(materials, material_names, colors):
            potentials = []
            
            for t in t_values:
                # 构建输入
                input_row = [
                    t,
                    base_scenario['n_e'],
                    base_scenario['T_e'],
                    base_scenario['n_i'],
                    base_scenario['T_i'],
                    base_scenario['S_flux'],
                    base_scenario['alpha_sun'],
                    mat_id
                ]
                
                inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
                
                # 模型预测
                with torch.no_grad():
                    prediction = self.model.predict(inputs).cpu().numpy().item()
                
                potentials.append(prediction)
            
            # 时间演化对比
            axes[0, 0].plot(t_values/3600, potentials, color=color, 
                          linewidth=2, label=mat_name)
        
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Surface Potential (V)')
        axes[0, 0].set_title('Material Comparison: Time Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 不同环境条件下的材料对比
        conditions = [
            {'name': 'Low Density', 'n_e': 1e5, 'T_e': 0.5},
            {'name': 'Medium Density', 'n_e': 1e6, 'T_e': 1.0},
            {'name': 'High Density', 'n_e': 1e7, 'T_e': 2.0}
        ]
        
        for i, condition in enumerate(conditions):
            ax = axes[0, 1] if i == 0 else (axes[1, 0] if i == 1 else axes[1, 1])
            
            for mat_id, mat_name, color in zip(materials, material_names, colors):
                potentials = []
                
                for t in t_values:
                    input_row = [
                        t,
                        condition['n_e'],
                        condition['T_e'],
                        condition['n_e'],  # n_i = n_e
                        condition['T_e'] * 0.1,  # T_i = T_e * 0.1
                        base_scenario['S_flux'],
                        base_scenario['alpha_sun'],
                        mat_id
                    ]
                    
                    inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        prediction = self.model.predict(inputs).cpu().numpy().item()
                    
                    potentials.append(prediction)
                
                ax.plot(t_values/3600, potentials, color=color, 
                       linewidth=2, label=mat_name)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Surface Potential (V)')
            ax.set_title(f'{condition["name"]} Environment')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Material comparison plot saved to {output_path}")
    
    def create_animation(self, scenario: Dict[str, Any],
                        output_path: str = 'charging_animation.gif') -> None:
        """
        创建充电过程动画
        
        Args:
            scenario: 场景参数
            output_path: 输出文件路径
        """
        logging.info("Creating charging animation...")
        
        # 生成时间序列
        t_end = scenario.get('t_end', 7200)
        t_values = np.linspace(0, t_end, 100)
        
        # 预计算所有时间点的结果
        potentials = []
        for t in t_values:
            input_row = [
                t,
                scenario['n_e'],
                scenario['T_e'],
                scenario['n_i'],
                scenario['T_i'],
                scenario['S_flux'],
                scenario['alpha_sun'],
                scenario['material_id']
            ]
            
            inputs = torch.tensor([input_row], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                prediction = self.model.predict(inputs).cpu().numpy().item()
            
            potentials.append(prediction)
        
        # 创建动画
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            current_time = t_values[frame]
            current_potential = potentials[frame]
            
            # 左图：时间演化曲线
            ax1.plot(t_values[:frame+1]/3600, potentials[:frame+1], 'b-', linewidth=2)
            ax1.scatter([current_time/3600], [current_potential], color='red', s=100, zorder=5)
            ax1.set_xlim(0, t_end/3600)
            ax1.set_ylim(min(potentials)*1.1, max(potentials)*1.1)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Surface Potential (V)')
            ax1.set_title('Charging Evolution')
            ax1.grid(True, alpha=0.3)
            
            # 右图：当前状态指示器
            ax2.bar(['Current\nPotential'], [current_potential], 
                   color='blue' if current_potential > 0 else 'red', alpha=0.7)
            ax2.set_ylim(min(potentials)*1.2, max(potentials)*1.2)
            ax2.set_ylabel('Potential (V)')
            ax2.set_title(f'Time: {current_time/3600:.2f} hours')
            ax2.grid(True, alpha=0.3)
            
            # 添加场景信息
            info_text = (
                f"n_e = {scenario['n_e']:.1e} m⁻³\n"
                f"T_e = {scenario['T_e']:.1f} eV\n"
                f"S_flux = {scenario['S_flux']:.0f} W/m²"
            )
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(t_values), 
                           interval=100, blit=False, repeat=True)
        
        # 保存动画
        anim.save(output_path, writer='pillow', fps=10)
        plt.close()
        
        logging.info(f"Animation saved to {output_path}")
    
    def generate_comprehensive_report(self, scenarios: List[Dict[str, Any]],
                                    output_dir: str = 'visualization_results') -> None:
        """
        生成综合可视化报告
        
        Args:
            scenarios: 场景列表
            output_dir: 输出目录
        """
        logging.info("Generating comprehensive visualization report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 时间演化图
        self.visualize_time_evolution(scenarios, 
                                    str(output_path / 'time_evolution.png'))
        
        # 2. 参数敏感性分析
        base_scenario = scenarios[0] if scenarios else {
            'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.1,
            'S_flux': 1361.0, 'alpha_sun': 0.0, 'material_id': 0
        }
        
        # 电子密度敏感性
        self.visualize_parameter_sensitivity(
            base_scenario, 'n_e', (1e4, 1e8),
            str(output_path / 'sensitivity_n_e.png')
        )
        
        # 电子温度敏感性
        self.visualize_parameter_sensitivity(
            base_scenario, 'T_e', (0.1, 5.0),
            str(output_path / 'sensitivity_T_e.png')
        )
        
        # 太阳辐射通量敏感性
        self.visualize_parameter_sensitivity(
            base_scenario, 'S_flux', (0, 2000),
            str(output_path / 'sensitivity_S_flux.png')
        )
        
        # 3. 相空间图
        if len(scenarios) > 10:
            self.visualize_phase_space(
                scenarios, 'n_e', 'T_e',
                str(output_path / 'phase_space_n_e_T_e.png')
            )
        
        # 4. 电流分量分析
        self.visualize_current_components(
            base_scenario,
            str(output_path / 'current_components.png')
        )
        
        # 5. 材料对比
        self.visualize_material_comparison(
            base_scenario,
            str(output_path / 'material_comparison.png')
        )
        
        # 6. 创建动画（可选）
        if self.viz_config.get('create_animation', False):
            self.create_animation(
                base_scenario,
                str(output_path / 'charging_animation.gif')
            )
        
        logging.info(f"Comprehensive visualization report generated in {output_path}")

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

def load_scenarios_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    从CSV文件加载场景
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        scenarios: 场景列表
    """
    df = pd.read_csv(csv_path)
    
    scenarios = []
    for _, row in df.iterrows():
        scenario = {
            'n_e': row['n_e'],
            'T_e': row['T_e'],
            'n_i': row['n_i'],
            'T_i': row['T_i'],
            'S_flux': row['S_flux'],
            'alpha_sun': row['alpha_sun'],
            'material_id': int(row['material_id'])
        }
        scenarios.append(scenario)
    
    return scenarios

def create_default_scenarios() -> List[Dict[str, Any]]:
    """
    创建默认测试场景
    
    Returns:
        scenarios: 默认场景列表
    """
    scenarios = [
        {
            'name': 'Quiet Space Weather',
            'n_e': 1e5, 'T_e': 0.5, 'n_i': 1e5, 'T_i': 0.05,
            'S_flux': 1361.0, 'alpha_sun': 0.0, 'material_id': 0,
            't_end': 7200
        },
        {
            'name': 'Active Space Weather',
            'n_e': 1e7, 'T_e': 2.0, 'n_i': 1e7, 'T_i': 0.2,
            'S_flux': 1361.0, 'alpha_sun': 0.0, 'material_id': 0,
            't_end': 7200
        },
        {
            'name': 'Eclipse Transition',
            'n_e': 1e6, 'T_e': 1.0, 'n_i': 1e6, 'T_i': 0.1,
            'S_flux': 0.0, 'alpha_sun': 0.0, 'material_id': 0,
            't_end': 7200,
            'events': [{'time': 1800, 'label': 'Enter Eclipse'}]
        }
    ]
    
    return scenarios

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
            logging.FileHandler('visualization.log')
        ]
    )

def main():
    """
    主可视化函数
    """
    parser = argparse.ArgumentParser(description='Visualize PINN model results')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Trained model path')
    parser.add_argument('--scenarios', type=str, default=None,
                       help='Scenarios CSV file path (optional)')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device_config = config.get('device', {})
    if device_config.get('use_gpu', True) and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}")
        logging.info(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    # 创建可视化器
    visualizer = PINNVisualizer(config, args.model, device)
    
    # 加载场景
    if args.scenarios and Path(args.scenarios).exists():
        scenarios = load_scenarios_from_csv(args.scenarios)
        logging.info(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")
    else:
        scenarios = create_default_scenarios()
        logging.info(f"Using {len(scenarios)} default scenarios")
    
    # 生成综合可视化报告
    visualizer.generate_comprehensive_report(scenarios, args.output_dir)
    
    logging.info("Visualization completed successfully!")

if __name__ == '__main__':
    main()