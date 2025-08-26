"""评估脚本

航天器表面充电PINN模型的评估脚本。
包含以下功能：
1. 模型预测精度评估
2. 物理合理性核查
3. 不同材料对比分析
4. 特殊场景验证

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
import deepxde as dde
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from pinn_model.physics import create_physics_model
from pinn_model.networks import create_networks
from pinn_model.dataset import create_dataloader


class PINNEvaluator:
    """PINN模型评估器
    
    提供全面的模型评估功能，包括精度评估和物理核查。
    """
    
    def __init__(self, config: Dict, model_path: str, experiment_name: str = None):
        """初始化评估器
        
        Args:
            config: 配置字典
            model_path: 模型文件路径
            experiment_name: 实验名称
        """
        self.config = config
        self.model_path = model_path
        self.experiment_name = experiment_name or f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建结果目录
        self.results_dir = Path(config['evaluation']['results_dir']) / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.physics_model = create_physics_model(config)
        self.networks = create_networks(config)
        self.dataloader = create_dataloader(config)
        
        # 加载模型
        self._load_model()
        
        self.logger.info(f"Initialized PINN evaluator for: {self.experiment_name}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.results_dir / "evaluation.log"
        
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
            else:
                self.logger.warning(f"Network weights file not found: {networks_path}")
            
            # 设置为评估模式
            self.networks.main_net.eval()
            self.networks.yield_net.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_test_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载测试数据
        
        Args:
            data_path: 测试数据文件路径
            
        Returns:
            X_test: 测试输入数据
            y_test: 测试目标数据
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data not found: {data_path}")
        
        # 加载数据集
        self.dataloader.load_datasets(data_path)
        dataset_info = self.dataloader.get_dataset_info()
        
        self.logger.info(f"Loaded test data: {dataset_info}")
        
        # 获取测试数据
        test_dataset = self.dataloader.test_dataset
        X_test = test_dataset.inputs.numpy()
        y_test = test_dataset.targets.numpy()
        
        return X_test, y_test
    
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
    
    def evaluate_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估模型预测精度
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标数据
            
        Returns:
            metrics: 评估指标字典
        """
        self.logger.info("Evaluating model accuracy...")
        
        # 模型预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 相对误差
        relative_error = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))
        
        # 最大误差
        max_error = np.max(np.abs(y_test - y_pred))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Relative_Error': relative_error,
            'Max_Error': max_error
        }
        
        # 记录结果
        self.logger.info("Accuracy Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
        
        # 保存结果
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.results_dir / "accuracy_metrics.csv", index=False)
        
        return metrics
    
    def physics_sanity_check(self) -> Dict[str, bool]:
        """物理合理性核查
        
        Returns:
            check_results: 核查结果字典
        """
        self.logger.info("Performing physics sanity checks...")
        
        check_results = {}
        
        # 1. 地球阴影区测试
        shadow_result = self._test_earth_shadow_scenario()
        check_results['earth_shadow'] = shadow_result
        
        # 2. 平静光照区测试
        sunlight_result = self._test_quiet_sunlight_scenario()
        check_results['quiet_sunlight'] = sunlight_result
        
        # 3. 材料对比测试
        material_result = self._test_material_comparison()
        check_results['material_comparison'] = material_result
        
        # 4. 时间演化测试
        evolution_result = self._test_time_evolution()
        check_results['time_evolution'] = evolution_result
        
        # 记录结果
        self.logger.info("Physics Sanity Check Results:")
        for test, passed in check_results.items():
            status = "PASSED" if passed else "FAILED"
            self.logger.info(f"  {test}: {status}")
        
        # 保存结果
        results_df = pd.DataFrame([check_results])
        results_df.to_csv(self.results_dir / "physics_checks.csv", index=False)
        
        return check_results
    
    def _test_earth_shadow_scenario(self) -> bool:
        """测试地球阴影区场景
        
        验证当太阳辐射通量从正值突变为0时，表面电位是否迅速下降为负值。
        
        Returns:
            passed: 测试是否通过
        """
        self.logger.info("Testing Earth shadow scenario...")
        
        # 创建测试数据：太阳辐射通量从正值变为0
        t_points = np.linspace(0, 3600, 100)  # 1小时
        n_e = 1e6  # 电子密度
        T_e = 1.0  # 电子温度 (eV)
        n_i = 1e6  # 离子密度
        T_i = 0.1  # 离子温度 (eV)
        alpha_sun = 0.0  # 垂直入射
        material_id = 0  # Kapton
        
        # 光照区和阴影区
        S_flux_sunlight = 1361.0  # 太阳常数
        S_flux_shadow = 0.0  # 阴影区
        
        # 光照区预测
        X_sunlight = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux_sunlight), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        # 阴影区预测
        X_shadow = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux_shadow), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        V_sunlight = self.predict(X_sunlight)
        V_shadow = self.predict(X_shadow)
        
        # 保存结果
        shadow_data = pd.DataFrame({
            'time': t_points,
            'V_sunlight': V_sunlight.flatten(),
            'V_shadow': V_shadow.flatten()
        })
        shadow_data.to_csv(self.results_dir / "earth_shadow_test.csv", index=False)
        
        # 检查：阴影区电位应该比光照区更负
        mean_V_sunlight = np.mean(V_sunlight)
        mean_V_shadow = np.mean(V_shadow)
        
        passed = mean_V_shadow < mean_V_sunlight - 1.0  # 至少相差1V
        
        self.logger.info(f"  Mean V_sunlight: {mean_V_sunlight:.3f} V")
        self.logger.info(f"  Mean V_shadow: {mean_V_shadow:.3f} V")
        
        return passed
    
    def _test_quiet_sunlight_scenario(self) -> bool:
        """测试平静光照区场景
        
        验证在平静空间天气条件下，表面电位是否稳定在微弱正值。
        
        Returns:
            passed: 测试是否通过
        """
        self.logger.info("Testing quiet sunlight scenario...")
        
        # 平静空间天气参数
        t_points = np.linspace(0, 7200, 100)  # 2小时
        n_e = 5e5  # 较低电子密度
        T_e = 0.5  # 较低电子温度
        n_i = 5e5  # 较低离子密度
        T_i = 0.05  # 较低离子温度
        S_flux = 1361.0  # 标准太阳辐射
        alpha_sun = 0.0  # 垂直入射
        material_id = 0  # Kapton
        
        X_quiet = np.column_stack([
            t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
            np.full_like(t_points, n_i), np.full_like(t_points, T_i),
            np.full_like(t_points, S_flux), np.full_like(t_points, alpha_sun),
            np.full_like(t_points, material_id)
        ])
        
        V_quiet = self.predict(X_quiet)
        
        # 保存结果
        quiet_data = pd.DataFrame({
            'time': t_points,
            'V_quiet': V_quiet.flatten()
        })
        quiet_data.to_csv(self.results_dir / "quiet_sunlight_test.csv", index=False)
        
        # 检查：电位应该在0-10V之间且相对稳定
        mean_V = np.mean(V_quiet)
        std_V = np.std(V_quiet)
        
        passed = (0 < mean_V < 10.0) and (std_V < 2.0)
        
        self.logger.info(f"  Mean V_quiet: {mean_V:.3f} V")
        self.logger.info(f"  Std V_quiet: {std_V:.3f} V")
        
        return passed
    
    def _test_material_comparison(self) -> bool:
        """测试不同材料对比
        
        验证不同材料的充电行为是否符合物理预期。
        
        Returns:
            passed: 测试是否通过
        """
        self.logger.info("Testing material comparison...")
        
        # 标准测试条件
        t_points = np.linspace(0, 3600, 50)
        n_e = 1e6
        T_e = 1.0
        n_i = 1e6
        T_i = 0.1
        S_flux = 1361.0
        alpha_sun = 0.0
        
        materials = self.config['data_generation']['materials']
        material_results = {}
        
        for i, material in enumerate(materials):
            X_material = np.column_stack([
                t_points, np.full_like(t_points, n_e), np.full_like(t_points, T_e),
                np.full_like(t_points, n_i), np.full_like(t_points, T_i),
                np.full_like(t_points, S_flux), np.full_like(t_points, alpha_sun),
                np.full_like(t_points, i)
            ])
            
            V_material = self.predict(X_material)
            material_results[material] = np.mean(V_material)
        
        # 保存结果
        material_data = pd.DataFrame([
            {'material': mat, 'mean_voltage': V} 
            for mat, V in material_results.items()
        ])
        material_data.to_csv(self.results_dir / "material_comparison_test.csv", index=False)
        
        # 检查：不同材料应该有不同的充电行为
        voltages = list(material_results.values())
        voltage_range = max(voltages) - min(voltages)
        
        passed = voltage_range > 1.0  # 材料间电位差应大于1V
        
        self.logger.info("  Material comparison results:")
        for material, voltage in material_results.items():
            self.logger.info(f"    {material}: {voltage:.3f} V")
        
        return passed
    
    def _test_time_evolution(self) -> bool:
        """测试时间演化
        
        验证充电过程的时间演化是否合理。
        
        Returns:
            passed: 测试是否通过
        """
        self.logger.info("Testing time evolution...")
        
        # 长时间演化测试
        t_points = np.linspace(0, 10800, 200)  # 3小时
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
        
        # 保存结果
        evolution_data = pd.DataFrame({
            'time': t_points,
            'voltage': V_evolution.flatten()
        })
        evolution_data.to_csv(self.results_dir / "time_evolution_test.csv", index=False)
        
        # 检查：电位应该趋于稳定（后期变化率小）
        dV_dt = np.gradient(V_evolution.flatten(), t_points)
        final_rate = np.mean(np.abs(dV_dt[-20:]))  # 最后20个点的平均变化率
        
        passed = final_rate < 1e-4  # 变化率应小于0.0001 V/s
        
        self.logger.info(f"  Final evolution rate: {final_rate:.6f} V/s")
        
        return passed
    
    def generate_evaluation_report(self, accuracy_metrics: Dict, physics_checks: Dict):
        """生成评估报告
        
        Args:
            accuracy_metrics: 精度评估结果
            physics_checks: 物理核查结果
        """
        self.logger.info("Generating evaluation report...")
        
        report_path = self.results_dir / "evaluation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# PINN Model Evaluation Report\n\n")
            f.write(f"**Experiment:** {self.experiment_name}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.model_path}\n\n")
            
            # 精度评估
            f.write("## Accuracy Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in accuracy_metrics.items():
                f.write(f"| {metric} | {value:.6f} |\n")
            f.write("\n")
            
            # 物理核查
            f.write("## Physics Sanity Checks\n\n")
            f.write("| Test | Result |\n")
            f.write("|------|--------|\n")
            for test, passed in physics_checks.items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                f.write(f"| {test} | {status} |\n")
            f.write("\n")
            
            # 总结
            total_checks = len(physics_checks)
            passed_checks = sum(physics_checks.values())
            f.write("## Summary\n\n")
            f.write(f"- **Accuracy:** R² = {accuracy_metrics['R2']:.4f}\n")
            f.write(f"- **Physics Checks:** {passed_checks}/{total_checks} passed\n")
            
            if passed_checks == total_checks:
                f.write(f"- **Overall Status:** ✅ All tests passed\n")
            else:
                f.write(f"- **Overall Status:** ⚠️ Some tests failed\n")
        
        self.logger.info(f"Evaluation report saved to: {report_path}")
    
    def evaluate(self, test_data_path: str = None) -> Dict:
        """完整评估流程
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            results: 评估结果字典
        """
        self.logger.info("Starting model evaluation...")
        
        results = {}
        
        # 1. 精度评估
        if test_data_path and os.path.exists(test_data_path):
            X_test, y_test = self.load_test_data(test_data_path)
            accuracy_metrics = self.evaluate_accuracy(X_test, y_test)
            results['accuracy'] = accuracy_metrics
        else:
            self.logger.warning("No test data provided, skipping accuracy evaluation")
            results['accuracy'] = {}
        
        # 2. 物理核查
        physics_checks = self.physics_sanity_check()
        results['physics'] = physics_checks
        
        # 3. 生成报告
        self.generate_evaluation_report(results['accuracy'], results['physics'])
        
        self.logger.info("Evaluation completed!")
        
        return results


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
    parser = argparse.ArgumentParser(description='Evaluate spacecraft charging PINN model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Trained model path')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Test data file path (optional)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建评估器
    evaluator = PINNEvaluator(config, args.model, args.experiment)
    
    # 开始评估
    try:
        results = evaluator.evaluate(args.test_data)
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {evaluator.results_dir}")
        
        # 打印简要结果
        if 'accuracy' in results and results['accuracy']:
            print(f"\nAccuracy: R² = {results['accuracy']['R2']:.4f}")
        
        if 'physics' in results:
            passed = sum(results['physics'].values())
            total = len(results['physics'])
            print(f"Physics checks: {passed}/{total} passed")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()