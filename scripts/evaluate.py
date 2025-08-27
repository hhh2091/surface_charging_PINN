# 评估脚本
# 实现模型性能评估和物理核查

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.networks import create_pinn_model
from pinn_model.dataset import create_data_loader
from pinn_model.physics import PhysicsModel

class PINNEvaluator:
    """
    PINN模型评估器
    
    功能：
    1. 模型性能评估
    2. 物理核查
    3. 结果可视化
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str, device: torch.device):
        """
        初始化评估器
        
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
        
        # 评估结果
        self.evaluation_results = {}
        
        logging.info(f"PINN evaluator initialized on device: {device}")
    
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
    
    def evaluate_on_dataset(self, data_loader, dataset_name: str = 'test') -> Dict[str, float]:
        """
        在数据集上评估模型性能
        
        Args:
            data_loader: 数据加载器
            dataset_name: 数据集名称
            
        Returns:
            metrics: 评估指标字典
        """
        logging.info(f"Evaluating on {dataset_name} dataset...")
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['outputs'].to(self.device)
                
                # 模型预测
                predictions = self.model.predict(inputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_inputs.append(inputs.cpu().numpy())
        
        # 合并所有批次
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 相对误差
        relative_error = np.abs((predictions - targets) / (targets + 1e-8))
        mean_relative_error = np.mean(relative_error)
        max_relative_error = np.max(relative_error)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_relative_error': mean_relative_error,
            'max_relative_error': max_relative_error,
            'num_samples': len(predictions)
        }
        
        # 保存详细结果
        self.evaluation_results[dataset_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'inputs': inputs
        }
        
        logging.info(f"{dataset_name.capitalize()} dataset evaluation completed:")
        logging.info(f"  RMSE: {rmse:.6f}")
        logging.info(f"  MAE: {mae:.6f}")
        logging.info(f"  R²: {r2:.6f}")
        logging.info(f"  Mean Relative Error: {mean_relative_error:.6f}")
        
        return metrics
    
    def physics_sanity_check(self) -> Dict[str, Any]:
        """
        执行物理核查
        
        Returns:
            sanity_results: 物理核查结果
        """
        logging.info("Performing physics sanity checks...")
        
        sanity_results = {}
        
        # 1. 地球阴影区测试
        sanity_results['shadow_test'] = self._test_shadow_transition()
        
        # 2. 平静空间天气测试
        sanity_results['quiet_space_test'] = self._test_quiet_space_weather()
        
        # 3. 材料对比测试
        sanity_results['material_comparison'] = self._test_material_comparison()
        
        # 4. 物理残差检查
        sanity_results['physics_residual'] = self._check_physics_residual()
        
        # 5. 初始条件检查
        sanity_results['initial_condition'] = self._check_initial_condition()
        
        return sanity_results
    
    def _test_shadow_transition(self) -> Dict[str, Any]:
        """
        测试进入地球阴影区的场景
        
        Returns:
            shadow_results: 阴影区测试结果
        """
        logging.info("Testing shadow transition scenario...")
        
        # 创建测试场景：从光照区进入阴影区
        t_values = np.linspace(0, 3600, 100)  # 1小时
        
        # 环境参数（典型地球轨道）
        n_e = 1e6  # 电子密度 (m^-3)
        T_e = 1.0  # 电子温度 (eV)
        n_i = 1e6  # 离子密度 (m^-3)
        T_i = 0.1  # 离子温度 (eV)
        alpha_sun = 0.0  # 垂直入射
        material_id = 0  # Kapton
        
        # 太阳辐射通量：在t=1800s时从光照进入阴影
        S_flux_values = np.where(t_values < 1800, 1361.0, 0.0)
        
        # 构建输入数据
        inputs = []
        for i, t in enumerate(t_values):
            inputs.append([t, n_e, T_e, n_i, T_i, S_flux_values[i], alpha_sun, material_id])
        
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(inputs).cpu().numpy().flatten()
        
        # 分析结果
        sunlight_potential = predictions[t_values < 1800]
        shadow_potential = predictions[t_values >= 1800]
        
        # 检查物理合理性
        sunlight_mean = np.mean(sunlight_potential[-10:])  # 进入阴影前的稳态
        shadow_final = shadow_potential[-1]  # 阴影区最终状态
        
        potential_drop = sunlight_mean - shadow_final
        is_physically_reasonable = (
            sunlight_mean > 0 and  # 光照区应为正电位
            shadow_final < 0 and   # 阴影区应为负电位
            potential_drop > 1.0   # 电位下降应显著
        )
        
        shadow_results = {
            'sunlight_mean_potential': sunlight_mean,
            'shadow_final_potential': shadow_final,
            'potential_drop': potential_drop,
            'is_physically_reasonable': is_physically_reasonable,
            'time_values': t_values,
            'predictions': predictions,
            'S_flux_values': S_flux_values
        }
        
        logging.info(f"Shadow transition test:")
        logging.info(f"  Sunlight potential: {sunlight_mean:.3f} V")
        logging.info(f"  Shadow potential: {shadow_final:.3f} V")
        logging.info(f"  Potential drop: {potential_drop:.3f} V")
        logging.info(f"  Physically reasonable: {is_physically_reasonable}")
        
        return shadow_results
    
    def _test_quiet_space_weather(self) -> Dict[str, Any]:
        """
        测试平静空间天气下的光照区场景
        
        Returns:
            quiet_results: 平静空间天气测试结果
        """
        logging.info("Testing quiet space weather scenario...")
        
        # 创建测试场景：平静空间天气，持续光照
        t_values = np.linspace(0, 7200, 100)  # 2小时
        
        # 平静空间天气参数
        n_e = 1e5   # 较低电子密度
        T_e = 0.5   # 较低电子温度
        n_i = 1e5   # 较低离子密度
        T_i = 0.05  # 较低离子温度
        S_flux = 1361.0  # 标准太阳辐射
        alpha_sun = 0.0  # 垂直入射
        material_id = 0  # Kapton
        
        # 构建输入数据
        inputs = []
        for t in t_values:
            inputs.append([t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id])
        
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(inputs).cpu().numpy().flatten()
        
        # 分析结果
        final_potential = predictions[-1]
        potential_stability = np.std(predictions[-20:])  # 最后20个点的标准差
        
        # 检查物理合理性
        is_physically_reasonable = (
            0.1 < final_potential < 10.0 and  # 应为小的正电位
            potential_stability < 0.5          # 应相对稳定
        )
        
        quiet_results = {
            'final_potential': final_potential,
            'potential_stability': potential_stability,
            'is_physically_reasonable': is_physically_reasonable,
            'time_values': t_values,
            'predictions': predictions
        }
        
        logging.info(f"Quiet space weather test:")
        logging.info(f"  Final potential: {final_potential:.3f} V")
        logging.info(f"  Potential stability (std): {potential_stability:.3f} V")
        logging.info(f"  Physically reasonable: {is_physically_reasonable}")
        
        return quiet_results
    
    def _test_material_comparison(self) -> Dict[str, Any]:
        """
        测试不同材料的充电行为对比
        
        Returns:
            material_results: 材料对比测试结果
        """
        logging.info("Testing material comparison...")
        
        # 创建测试场景：相同环境条件，不同材料
        t_values = np.linspace(0, 3600, 50)  # 1小时
        
        # 环境参数
        n_e = 5e5
        T_e = 1.0
        n_i = 5e5
        T_i = 0.1
        S_flux = 1361.0
        alpha_sun = 0.0
        
        materials = [0, 1]  # Kapton, Aluminum
        material_names = ['Kapton', 'Aluminum']
        material_results = {}
        
        for mat_id, mat_name in zip(materials, material_names):
            # 构建输入数据
            inputs = []
            for t in t_values:
                inputs.append([t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, mat_id])
            
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
            # 模型预测
            self.model.eval()
            with torch.no_grad():
                predictions = self.model.predict(inputs).cpu().numpy().flatten()
            
            material_results[mat_name] = {
                'final_potential': predictions[-1],
                'max_potential': np.max(predictions),
                'min_potential': np.min(predictions),
                'time_values': t_values,
                'predictions': predictions
            }
        
        # 分析材料差异
        kapton_final = material_results['Kapton']['final_potential']
        aluminum_final = material_results['Aluminum']['final_potential']
        potential_difference = abs(kapton_final - aluminum_final)
        
        # 检查物理合理性（不同材料应有不同的充电行为）
        is_physically_reasonable = potential_difference > 0.1
        
        comparison_results = {
            'materials': material_results,
            'potential_difference': potential_difference,
            'is_physically_reasonable': is_physically_reasonable
        }
        
        logging.info(f"Material comparison test:")
        logging.info(f"  Kapton final potential: {kapton_final:.3f} V")
        logging.info(f"  Aluminum final potential: {aluminum_final:.3f} V")
        logging.info(f"  Potential difference: {potential_difference:.3f} V")
        logging.info(f"  Physically reasonable: {is_physically_reasonable}")
        
        return comparison_results
    
    def _check_physics_residual(self) -> Dict[str, Any]:
        """
        检查物理残差
        
        Returns:
            residual_results: 物理残差检查结果
        """
        logging.info("Checking physics residual...")
        
        # 生成随机测试点
        num_samples = 1000
        
        # 参数范围（从配置文件获取）
        data_config = self.config['data_generation']
        param_ranges = data_config['parameter_ranges']
        
        # 生成随机输入
        inputs = []
        for _ in range(num_samples):
            t = np.random.uniform(0, data_config['time_domain']['t_end'])
            n_e = np.random.uniform(param_ranges['n_e']['min'], param_ranges['n_e']['max'])
            T_e = np.random.uniform(param_ranges['T_e']['min'], param_ranges['T_e']['max'])
            n_i = np.random.uniform(param_ranges['n_i']['min'], param_ranges['n_i']['max'])
            T_i = np.random.uniform(param_ranges['T_i']['min'], param_ranges['T_i']['max'])
            S_flux = np.random.uniform(param_ranges['S_flux']['min'], param_ranges['S_flux']['max'])
            alpha_sun = np.random.uniform(param_ranges['alpha_sun']['min'], param_ranges['alpha_sun']['max'])
            material_id = np.random.choice([0, 1])
            
            inputs.append([t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id])
        
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        inputs.requires_grad_(True)
        
        # 计算物理残差
        self.model.eval()
        residuals = self.model.compute_physics_residual(inputs)
        residuals_np = residuals.detach().cpu().numpy()
        
        # 统计分析
        mean_residual = np.mean(np.abs(residuals_np))
        max_residual = np.max(np.abs(residuals_np))
        std_residual = np.std(residuals_np)
        
        # 检查物理残差是否足够小
        is_acceptable = mean_residual < 1e-3 and max_residual < 1e-2
        
        residual_results = {
            'mean_absolute_residual': mean_residual,
            'max_absolute_residual': max_residual,
            'std_residual': std_residual,
            'is_acceptable': is_acceptable,
            'residuals': residuals_np
        }
        
        logging.info(f"Physics residual check:")
        logging.info(f"  Mean absolute residual: {mean_residual:.6f}")
        logging.info(f"  Max absolute residual: {max_residual:.6f}")
        logging.info(f"  Std residual: {std_residual:.6f}")
        logging.info(f"  Acceptable: {is_acceptable}")
        
        return residual_results
    
    def _check_initial_condition(self) -> Dict[str, Any]:
        """
        检查初始条件
        
        Returns:
            initial_results: 初始条件检查结果
        """
        logging.info("Checking initial condition...")
        
        # 生成t=0的测试点
        num_samples = 100
        
        # 参数范围
        data_config = self.config['data_generation']
        param_ranges = data_config['parameter_ranges']
        
        # 生成随机初始条件
        inputs = []
        for _ in range(num_samples):
            t = 0.0  # 初始时间
            n_e = np.random.uniform(param_ranges['n_e']['min'], param_ranges['n_e']['max'])
            T_e = np.random.uniform(param_ranges['T_e']['min'], param_ranges['T_e']['max'])
            n_i = np.random.uniform(param_ranges['n_i']['min'], param_ranges['n_i']['max'])
            T_i = np.random.uniform(param_ranges['T_i']['min'], param_ranges['T_i']['max'])
            S_flux = np.random.uniform(param_ranges['S_flux']['min'], param_ranges['S_flux']['max'])
            alpha_sun = np.random.uniform(param_ranges['alpha_sun']['min'], param_ranges['alpha_sun']['max'])
            material_id = np.random.choice([0, 1])
            
            inputs.append([t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id])
        
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(inputs).cpu().numpy().flatten()
        
        # 检查初始条件：V(t=0) = 0
        mean_initial_potential = np.mean(predictions)
        max_initial_potential = np.max(np.abs(predictions))
        
        # 检查是否满足初始条件
        is_satisfied = max_initial_potential < 0.1  # 允许小的误差
        
        initial_results = {
            'mean_initial_potential': mean_initial_potential,
            'max_absolute_initial_potential': max_initial_potential,
            'is_satisfied': is_satisfied,
            'predictions': predictions
        }
        
        logging.info(f"Initial condition check:")
        logging.info(f"  Mean initial potential: {mean_initial_potential:.6f} V")
        logging.info(f"  Max absolute initial potential: {max_initial_potential:.6f} V")
        logging.info(f"  Initial condition satisfied: {is_satisfied}")
        
        return initial_results
    
    def plot_evaluation_results(self, output_dir: str = 'evaluation_results') -> None:
        """
        绘制评估结果
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 预测vs真实值散点图
        self._plot_prediction_scatter(output_path)
        
        # 2. 物理核查结果图
        self._plot_physics_sanity_checks(output_path)
        
        # 3. 残差分布图
        self._plot_residual_distribution(output_path)
        
        # 4. 误差分析图
        self._plot_error_analysis(output_path)
        
        logging.info(f"Evaluation plots saved to {output_path}")
    
    def _plot_prediction_scatter(self, output_path: Path) -> None:
        """
        绘制预测vs真实值散点图
        
        Args:
            output_path: 输出路径
        """
        if 'test' not in self.evaluation_results:
            return
        
        results = self.evaluation_results['test']
        predictions = results['predictions'].flatten()
        targets = results['targets'].flatten()
        
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.scatter(targets, predictions, alpha=0.6, s=20)
        
        # 理想线
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # 设置图形
        plt.xlabel('True Values (V)')
        plt.ylabel('Predicted Values (V)')
        plt.title('Prediction vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        r2 = results['metrics']['r2']
        rmse = results['metrics']['rmse']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_physics_sanity_checks(self, output_path: Path) -> None:
        """
        绘制物理核查结果
        
        Args:
            output_path: 输出路径
        """
        if not hasattr(self, 'sanity_results'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 阴影区转换测试
        if 'shadow_test' in self.sanity_results:
            shadow_data = self.sanity_results['shadow_test']
            ax = axes[0, 0]
            
            t_vals = shadow_data['time_values'] / 3600  # 转换为小时
            predictions = shadow_data['predictions']
            S_flux = shadow_data['S_flux_values']
            
            # 电位曲线
            ax.plot(t_vals, predictions, 'b-', linewidth=2, label='Surface Potential')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Potential (V)')
            ax.set_title('Shadow Transition Test')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加阴影区标记
            shadow_start = 1800 / 3600
            ax.axvline(x=shadow_start, color='r', linestyle='--', alpha=0.7, label='Enter Shadow')
            ax.fill_between(t_vals[t_vals >= shadow_start], 
                          ax.get_ylim()[0], ax.get_ylim()[1], 
                          alpha=0.2, color='gray', label='Shadow Region')
        
        # 2. 平静空间天气测试
        if 'quiet_space_test' in self.sanity_results:
            quiet_data = self.sanity_results['quiet_space_test']
            ax = axes[0, 1]
            
            t_vals = quiet_data['time_values'] / 3600
            predictions = quiet_data['predictions']
            
            ax.plot(t_vals, predictions, 'g-', linewidth=2)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Potential (V)')
            ax.set_title('Quiet Space Weather Test')
            ax.grid(True, alpha=0.3)
        
        # 3. 材料对比测试
        if 'material_comparison' in self.sanity_results:
            material_data = self.sanity_results['material_comparison']['materials']
            ax = axes[1, 0]
            
            for mat_name, data in material_data.items():
                t_vals = data['time_values'] / 3600
                predictions = data['predictions']
                ax.plot(t_vals, predictions, linewidth=2, label=mat_name)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Potential (V)')
            ax.set_title('Material Comparison Test')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. 物理残差分布
        if 'physics_residual' in self.sanity_results:
            residual_data = self.sanity_results['physics_residual']
            ax = axes[1, 1]
            
            residuals = residual_data['residuals']
            ax.hist(residuals, bins=50, alpha=0.7, density=True)
            ax.set_xlabel('Physics Residual')
            ax.set_ylabel('Density')
            ax.set_title('Physics Residual Distribution')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_res = residual_data['mean_absolute_residual']
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            ax.text(0.05, 0.95, f'Mean |Residual| = {mean_res:.2e}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'physics_sanity_checks.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_distribution(self, output_path: Path) -> None:
        """
        绘制残差分布图
        
        Args:
            output_path: 输出路径
        """
        if 'test' not in self.evaluation_results:
            return
        
        results = self.evaluation_results['test']
        predictions = results['predictions'].flatten()
        targets = results['targets'].flatten()
        residuals = predictions - targets
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差直方图
        axes[0].hist(residuals, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[0].set_xlabel('Residuals (V)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Residual Distribution')
        axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[0].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[0].text(0.05, 0.95, f'Mean = {mean_res:.4f}\nStd = {std_res:.4f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self, output_path: Path) -> None:
        """
        绘制误差分析图
        
        Args:
            output_path: 输出路径
        """
        if 'test' not in self.evaluation_results:
            return
        
        results = self.evaluation_results['test']
        predictions = results['predictions'].flatten()
        targets = results['targets'].flatten()
        inputs = results['inputs']
        
        # 计算相对误差
        relative_errors = np.abs((predictions - targets) / (targets + 1e-8))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 误差vs真实值
        axes[0, 0].scatter(targets, relative_errors, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('True Values (V)')
        axes[0, 0].set_ylabel('Relative Error')
        axes[0, 0].set_title('Relative Error vs True Values')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差vs时间
        time_values = inputs[:, 0]
        axes[0, 1].scatter(time_values, relative_errors, alpha=0.6, s=20)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Relative Error')
        axes[0, 1].set_title('Relative Error vs Time')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 误差vs电子密度
        n_e_values = inputs[:, 1]
        axes[1, 0].scatter(n_e_values, relative_errors, alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Electron Density (m⁻³)')
        axes[1, 0].set_ylabel('Relative Error')
        axes[1, 0].set_title('Relative Error vs Electron Density')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差vs太阳辐射通量
        S_flux_values = inputs[:, 5]
        axes[1, 1].scatter(S_flux_values, relative_errors, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Solar Flux (W/m²)')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_title('Relative Error vs Solar Flux')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_report(self, output_path: str = 'evaluation_report.txt') -> None:
        """
        保存评估报告
        
        Args:
            output_path: 报告文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("PINN Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 数据集评估结果
            for dataset_name, results in self.evaluation_results.items():
                f.write(f"{dataset_name.upper()} Dataset Evaluation:\n")
                f.write("-" * 30 + "\n")
                
                metrics = results['metrics']
                f.write(f"Number of samples: {metrics['num_samples']}\n")
                f.write(f"RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"MAE: {metrics['mae']:.6f}\n")
                f.write(f"R²: {metrics['r2']:.6f}\n")
                f.write(f"Mean Relative Error: {metrics['mean_relative_error']:.6f}\n")
                f.write(f"Max Relative Error: {metrics['max_relative_error']:.6f}\n\n")
            
            # 物理核查结果
            if hasattr(self, 'sanity_results'):
                f.write("Physics Sanity Check Results:\n")
                f.write("-" * 30 + "\n")
                
                for test_name, results in self.sanity_results.items():
                    f.write(f"{test_name.replace('_', ' ').title()}:\n")
                    
                    if test_name == 'shadow_test':
                        f.write(f"  Sunlight potential: {results['sunlight_mean_potential']:.3f} V\n")
                        f.write(f"  Shadow potential: {results['shadow_final_potential']:.3f} V\n")
                        f.write(f"  Potential drop: {results['potential_drop']:.3f} V\n")
                        f.write(f"  Physically reasonable: {results['is_physically_reasonable']}\n")
                    
                    elif test_name == 'quiet_space_test':
                        f.write(f"  Final potential: {results['final_potential']:.3f} V\n")
                        f.write(f"  Potential stability: {results['potential_stability']:.3f} V\n")
                        f.write(f"  Physically reasonable: {results['is_physically_reasonable']}\n")
                    
                    elif test_name == 'material_comparison':
                        f.write(f"  Potential difference: {results['potential_difference']:.3f} V\n")
                        f.write(f"  Physically reasonable: {results['is_physically_reasonable']}\n")
                    
                    elif test_name == 'physics_residual':
                        f.write(f"  Mean absolute residual: {results['mean_absolute_residual']:.2e}\n")
                        f.write(f"  Max absolute residual: {results['max_absolute_residual']:.2e}\n")
                        f.write(f"  Acceptable: {results['is_acceptable']}\n")
                    
                    elif test_name == 'initial_condition':
                        f.write(f"  Mean initial potential: {results['mean_initial_potential']:.6f} V\n")
                        f.write(f"  Max absolute initial potential: {results['max_absolute_initial_potential']:.6f} V\n")
                        f.write(f"  Initial condition satisfied: {results['is_satisfied']}\n")
                    
                    f.write("\n")
        
        logging.info(f"Evaluation report saved to {output_path}")

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
            logging.FileHandler('evaluation.log')
        ]
    )

def main():
    """
    主评估函数
    """
    parser = argparse.ArgumentParser(description='Evaluate PINN model for spacecraft charging')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Trained model path')
    parser.add_argument('--test_data', type=str, default='data/test_data.csv',
                       help='Test data path')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
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
    
    # 创建评估器
    evaluator = PINNEvaluator(config, args.model, device)
    
    # 创建数据加载器
    data_loader = create_data_loader(config)
    
    # 加载测试数据
    if Path(args.test_data).exists():
        data_loader.load_datasets(test_path=args.test_data)
        data_loader.create_data_loaders()
        
        # 在测试集上评估
        if data_loader.test_loader is not None:
            evaluator.evaluate_on_dataset(data_loader.test_loader, 'test')
    else:
        logging.warning(f"Test data file not found: {args.test_data}")
    
    # 执行物理核查
    evaluator.sanity_results = evaluator.physics_sanity_check()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制评估结果
    evaluator.plot_evaluation_results(args.output_dir)
    
    # 保存评估报告
    report_path = output_dir / 'evaluation_report.txt'
    evaluator.save_evaluation_report(str(report_path))
    
    logging.info("Evaluation completed successfully!")

if __name__ == '__main__':
    main()