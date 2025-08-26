"""可视化脚本

航天器表面充电PINN模型的结果可视化脚本。
提供以下可视化功能：
1. 训练损失曲线
2. 模型预测vs真实值对比
3. 物理场景分析
4. 材料对比分析
5. 时间演化可视化
6. 参数敏感性分析

Author: PINN Engineering Team
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.physics import create_physics_model
from pinn_model.networks import create_networks
from pinn_model.dataset import create_dataloader

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class PINNVisualizer:
    """PINN模型可视化器
    
    提供全面的可视化功能，包括训练过程、预测结果和物理分析。
    """
    
    def __init__(self, config: Dict, model_path: str, experiment_name: str = None):
        """初始化可视化器
        
        Args:
            config: 配置字典
            model_path: 模型文件路径
            experiment_name: 实验名称
        """
        self.config = config
        self.model_path = model_path
        self.experiment_name = experiment_name or f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['visualization']['output_dir']) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.physics_model = create_physics_model(config)
        self.networks = create_networks(config)
        self.dataloader = create_dataloader(config)
        
        # 加载模型
        self._load_model()
        
        self.logger.info(f"Initialized PINN visualizer for: {self.experiment_name}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "visualization.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            # 加载网络权重
            networks_path = self.model_path.replace('.ckpt', '_networks.pth')
            if os.path.exists(networks_path):
                checkpoint = torch.load(networks_path, map_location=self.device)
                self.networks.main_net.load_state_dict(checkpoint['main_net_state_dict'])
                self.networks.yield_net.load_state_dict(checkpoint['yield_net_state_dict'])
                self.logger.info(f"Loaded network weights from {networks_path}")
            
            # 设置为评估模式
            self.networks.main_net.eval()
            self.networks.yield_net.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测
        
        Args:
            X: 输入数据 [N, 8]
            
        Returns:
            predictions: 预测结果 [N, 1]
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.networks.main_net(X_tensor)
            return predictions.cpu().numpy()
    
    def plot_training_history(self, history_file: str = None):
        """绘制训练历史
        
        Args:
            history_file: 训练历史文件路径
        """
        self.logger.info("Plotting training history...")
        
        # 如果没有提供历史文件，尝试从模型目录查找
        if history_file is None:
            model_dir = Path(self.model_path).parent
            history_file = model_dir / "training_history.csv"
        
        if not os.path.exists(history_file):
            self.logger.warning(f"Training history file not found: {history_file}")
            return
        
        # 读取训练历史
        history = pd.read_csv(history_file)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练历史', fontsize=16, fontweight='bold')
        
        # 总损失
        axes[0, 0].plot(history['epoch'], history['total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 物理残差损失
        axes[0, 1].plot(history['epoch'], history['residual_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('物理残差损失')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 数据损失
        if 'data_loss' in history.columns:
            axes[1, 0].plot(history['epoch'], history['data_loss'], 'g-', linewidth=2)
            axes[1, 0].set_title('数据损失')
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 初始条件损失
        if 'initial_loss' in history.columns:
            axes[1, 1].plot(history['epoch'], history['initial_loss'], 'm-', linewidth=2)
            axes[1, 1].set_title('初始条件损失')
            axes[1, 1].set_xlabel('训练轮次')
            axes[1, 1].set_ylabel('损失值')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved to {self.output_dir / 'training_history.png'}")
    
    def plot_prediction_comparison(self, test_data_path: str):
        """绘制预测vs真实值对比
        
        Args:
            test_data_path: 测试数据路径
        """
        self.logger.info("Plotting prediction comparison...")
        
        if not os.path.exists(test_data_path):
            self.logger.warning(f"Test data not found: {test_data_path}")
            return
        
        # 加载测试数据
        self.dataloader.load_datasets(test_data_path)
        test_dataset = self.dataloader.test_dataset
        X_test = test_dataset.inputs.numpy()
        y_test = test_dataset.targets.numpy()
        
        # 模型预测
        y_pred = self.predict(X_test)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 散点图
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=20)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        axes[0].set_xlabel('真实值 (V)')
        axes[0].set_ylabel('预测值 (V)')
        axes[0].set_title('预测vs真实值')
        axes[0].grid(True, alpha=0.3)
        
        # 计算R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 误差分布
        errors = y_pred.flatten() - y_test.flatten()
        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('预测误差 (V)')
        axes[1].set_ylabel('频次')
        axes[1].set_title('误差分布')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].grid(True, alpha=0.3)
        
        # 相对误差
        relative_errors = np.abs(errors) / (np.abs(y_test.flatten()) + 1e-8) * 100
        axes[2].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('相对误差 (%)')
        axes[2].set_ylabel('频次')
        axes[2].set_title('相对误差分布')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Prediction comparison plot saved to {self.output_dir / 'prediction_comparison.png'}")
    
    def plot_physics_scenarios(self):
        """绘制物理场景分析"""
        self.logger.info("Plotting physics scenarios...")
        
        # 1. 地球阴影区vs光照区
        self._plot_shadow_vs_sunlight()
        
        # 2. 材料对比
        self._plot_material_comparison()
        
        # 3. 时间演化
        self._plot_time_evolution()
        
        # 4. 参数敏感性
        self._plot_parameter_sensitivity()
    
    def _plot_shadow_vs_sunlight(self):
        """绘制阴影区vs光照区对比"""
        t_points = np.linspace(0, 7200, 100)  # 2小时
        
        # 标准环境参数
        n_e = 1e6
        T_e = 1.0
        n_i = 1e6
        T_i = 0.1
        alpha_sun = 0.0
        material_id = 0
        
        # 光照区和阴影区
        S_flux_sunlight = 1361.0
        S_flux_shadow = 0.0
        
        # 创建输入数据
        X_sunlight = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux_sunlight), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        X_shadow = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux_shadow), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        # 预测
        V_sunlight = self.predict(X_sunlight)
        V_shadow = self.predict(X_shadow)
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(t_points/3600, V_sunlight.flatten(), 'orange', linewidth=3, label='光照区')
        plt.plot(t_points/3600, V_shadow.flatten(), 'blue', linewidth=3, label='阴影区')
        plt.xlabel('时间 (小时)')
        plt.ylabel('表面电位 (V)')
        plt.title('地球阴影区vs光照区充电对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shadow_vs_sunlight.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_material_comparison(self):
        """绘制材料对比"""
        materials = self.config['data_generation']['materials']
        t_points = np.linspace(0, 3600, 50)
        
        # 标准环境参数
        n_e = 1e6
        T_e = 1.0
        n_i = 1e6
        T_i = 0.1
        S_flux = 1361.0
        alpha_sun = 0.0
        
        plt.figure(figsize=(12, 8))
        
        for i, material in enumerate(materials):
            X_material = np.column_stack([
                t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
                np.full_like(t_points, n_i), np.full_like(t_points, T_i),
                np.full_like(t_points, S_flux), np.full_like(t_points, alpha_sun),
                np.full_like(t_points, i)
            ])
            
            V_material = self.predict(X_material)
            plt.plot(t_points/60, V_material.flatten(), linewidth=3, label=material)
        
        plt.xlabel('时间 (分钟)')
        plt.ylabel('表面电位 (V)')
        plt.title('不同材料充电对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "material_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_evolution(self):
        """绘制时间演化"""
        t_points = np.linspace(0, 10800, 200)  # 3小时
        
        # 标准环境参数
        n_e = 1e6
        T_e = 1.0
        n_i = 1e6
        T_i = 0.1
        S_flux = 1361.0
        alpha_sun = 0.0
        material_id = 0
        
        X_evolution = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        V_evolution = self.predict(X_evolution)
        
        # 计算充电速率
        dV_dt = np.gradient(V_evolution.flatten(), t_points)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 电位演化
        axes[0].plot(t_points/3600, V_evolution.flatten(), 'b-', linewidth=3)
        axes[0].set_xlabel('时间 (小时)')
        axes[0].set_ylabel('表面电位 (V)')
        axes[0].set_title('表面电位时间演化')
        axes[0].grid(True, alpha=0.3)
        
        # 充电速率
        axes[1].plot(t_points/3600, dV_dt, 'r-', linewidth=2)
        axes[1].set_xlabel('时间 (小时)')
        axes[1].set_ylabel('充电速率 (V/s)')
        axes[1].set_title('充电速率时间演化')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self):
        """绘制参数敏感性分析"""
        # 基准参数
        base_params = {
            't': 1800,  # 30分钟
            'n_e': 1e6,
            'T_e': 1.0,
            'n_i': 1e6,
            'T_i': 0.1,
            'S_flux': 1361.0,
            'alpha_sun': 0.0,
            'material_id': 0
        }
        
        # 参数变化范围
        param_ranges = {
            'n_e': np.logspace(5, 7, 50),
            'T_e': np.linspace(0.1, 5.0, 50),
            'S_flux': np.linspace(0, 2000, 50),
            'alpha_sun': np.linspace(0, 90, 50)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = ['n_e', 'T_e', 'S_flux', 'alpha_sun']
        param_labels = ['电子密度 (m⁻³)', '电子温度 (eV)', '太阳辐射通量 (W/m²)', '入射角 (度)']
        
        for i, (param, label) in enumerate(zip(param_names, param_labels)):
            param_values = param_ranges[param]
            voltages = []
            
            for value in param_values:
                # 创建输入
                params = base_params.copy()
                params[param] = value
                
                X = np.array([[params['t'], params['n_e'], params['T_e'], params['n_i'], 
                             params['T_i'], params['S_flux'], params['alpha_sun'], params['material_id']]])
                
                V = self.predict(X)
                voltages.append(V[0, 0])
            
            axes[i].plot(param_values, voltages, 'b-', linewidth=3)
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('表面电位 (V)')
            axes[i].set_title(f'{label}敏感性')
            axes[i].grid(True, alpha=0.3)
            
            if param == 'n_e':
                axes[i].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        self.logger.info("Creating interactive dashboard...")
        
        # 创建参数控制面板
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间演化', '材料对比', '环境参数影响', '3D参数空间'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scatter3d"}]]
        )
        
        # 时间演化数据
        t_points = np.linspace(0, 7200, 100)
        base_X = np.column_stack([
            t_points, np.full_like(t_points, 1e6), np.full_like(t_points, 1.0),
            np.full_like(t_points, 1e6), np.full_like(t_points, 0.1),
            np.full_like(t_points, 1361.0), np.full_like(t_points, 0.0),
            np.full_like(t_points, 0)
        ])
        V_time = self.predict(base_X)
        
        fig.add_trace(
            go.Scatter(x=t_points/3600, y=V_time.flatten(), name='电位演化'),
            row=1, col=1
        )
        
        # 材料对比数据
        materials = self.config['data_generation']['materials']
        for i, material in enumerate(materials):
            material_X = base_X.copy()
            material_X[:, -1] = i
            V_material = self.predict(material_X)
            
            fig.add_trace(
                go.Scatter(x=t_points/3600, y=V_material.flatten(), name=material),
                row=1, col=2
            )
        
        # 保存交互式图表
        fig.update_layout(height=800, title_text="航天器表面充电PINN分析仪表板")
        fig.write_html(self.output_dir / "interactive_dashboard.html")
        
        self.logger.info(f"Interactive dashboard saved to {self.output_dir / 'interactive_dashboard.html'}")
    
    def generate_summary_report(self):
        """生成可视化总结报告"""
        self.logger.info("Generating summary report...")
        
        # 创建总结图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('航天器表面充电PINN模型分析总结', fontsize=20, fontweight='bold')
        
        # 这里可以添加各种总结性的可视化
        # 由于篇幅限制，这里只是示例框架
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary report saved to {self.output_dir / 'summary_report.png'}")
    
    def visualize_all(self, test_data_path: str = None, history_file: str = None):
        """执行所有可视化
        
        Args:
            test_data_path: 测试数据路径
            history_file: 训练历史文件路径
        """
        self.logger.info("Starting comprehensive visualization...")
        
        try:
            # 1. 训练历史
            if history_file:
                self.plot_training_history(history_file)
            
            # 2. 预测对比
            if test_data_path:
                self.plot_prediction_comparison(test_data_path)
            
            # 3. 物理场景分析
            self.plot_physics_scenarios()
            
            # 4. 交互式仪表板
            self.create_interactive_dashboard()
            
            # 5. 总结报告
            self.generate_summary_report()
            
            self.logger.info("All visualizations completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise


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
    parser = argparse.ArgumentParser(description='Visualize spacecraft charging PINN results')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Trained model path')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Test data file path (optional)')
    parser.add_argument('--history', type=str, default=None,
                       help='Training history file path (optional)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (optional)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新输出目录
    if args.output:
        config['visualization']['output_dir'] = args.output
    
    # 创建可视化器
    visualizer = PINNVisualizer(config, args.model, args.experiment)
    
    # 开始可视化
    try:
        visualizer.visualize_all(args.test_data, args.history)
        
        print("\nVisualization completed successfully!")
        print(f"Results saved to: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        raise


if __name__ == '__main__':
    main()