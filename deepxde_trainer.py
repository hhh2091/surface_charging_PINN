import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime

# 导入自定义模块
from deepxde_pinn_model import DeepXDESurfaceChargingPINN
from boundary_conditions import BoundaryConditionManager, StandardBoundaryConditions, InitialConditions, DataConstraintGenerator
from deepxde_loss_functions import AdaptiveLossWeighting, PINNLossMonitor, CustomLossFunction, LossScheduler
from deepxde_optimizer import TwoStageOptimizer, LearningRateScheduler, EarlyStopping

class DeepXDEPINNTrainer:
    """基于DeepXDE的PINN训练器"""
    
    def __init__(self, 
                 config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = self._setup_device()
        
        # 初始化组件
        self.pinn_model = None
        self.boundary_manager = None
        self.data_generator = None
        self.loss_weighting = None
        self.loss_monitor = None
        self.optimizer = None
        self.lr_scheduler = None
        self.early_stopping = None
        
        # 训练状态
        self.is_trained = False
        self.training_results = {}
        self.model_save_path = None
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', './results')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"DeepXDE PINN Trainer initialized")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("Using CPU")
        
        # 设置DeepXDE后端
        dde.config.set_default_float('float32')
        
        return device
    
    def setup_model(self):
        """设置PINN模型"""
        print("\nSetting up PINN model...")
        
        # 模型配置
        model_config = self.config['model']
        
        # 创建PINN模型
        self.pinn_model = DeepXDESurfaceChargingPINN(
            domain_bounds=model_config['domain_bounds'],
            physical_params=model_config['physical_params'],
            network_config=model_config['network_config']
        )
        
        print(f"PINN model created with {len(model_config['network_config']['hidden_layers'])} hidden layers")
        print(f"Domain bounds: {model_config['domain_bounds']}")
    
    def setup_boundary_conditions(self):
        """设置边界条件"""
        print("\nSetting up boundary conditions...")
        
        bc_config = self.config.get('boundary_conditions', {})
        domain_bounds = self.config['model']['domain_bounds']
        
        # 创建边界条件管理器
        self.boundary_manager = BoundaryConditionManager(domain_bounds)
        
        # 添加标准边界条件
        if bc_config.get('grounded_surface', False):
            boundary_func, value_func = StandardBoundaryConditions.grounded_surface(domain_bounds)
            self.boundary_manager.add_dirichlet_bc(
                boundary_func, value_func, "Grounded surface"
            )
        
        if 'floating_potential' in bc_config:
            V_float = bc_config['floating_potential']
            boundary_func, value_func = StandardBoundaryConditions.floating_potential(V_float)
            self.boundary_manager.add_dirichlet_bc(
                boundary_func, value_func, f"Floating potential: {V_float}V"
            )
        
        # 添加初始条件
        ic_config = bc_config.get('initial_conditions', {})
        if ic_config.get('type') == 'zero':
            initial_func = InitialConditions.zero_initial_potential()
            self.boundary_manager.add_initial_condition(
                initial_func, "Zero initial potential"
            )
        elif ic_config.get('type') == 'constant':
            V0 = ic_config.get('value', 0.0)
            initial_func = InitialConditions.constant_initial_potential(V0)
            self.boundary_manager.add_initial_condition(
                initial_func, f"Constant initial potential: {V0}V"
            )
        elif ic_config.get('type') == 'gaussian':
            initial_func = InitialConditions.gaussian_initial_potential(
                center=ic_config.get('center', [0.5, 0.5, 0.5]),
                amplitude=ic_config.get('amplitude', 1.0),
                width=ic_config.get('width', 0.1)
            )
            self.boundary_manager.add_initial_condition(
                initial_func, "Gaussian initial potential"
            )
        
        # 将边界条件添加到模型
        all_conditions = self.boundary_manager.get_all_conditions()
        if all_conditions:
            self.pinn_model.add_boundary_conditions(all_conditions)
            print(f"Added {len(all_conditions)} boundary/initial conditions")
    
    def setup_data_constraints(self):
        """设置数据约束"""
        print("\nSetting up data constraints...")
        
        data_config = self.config.get('data_constraints', {})
        
        if not data_config.get('enabled', False):
            print("Data constraints disabled")
            return
        
        domain_bounds = self.config['model']['domain_bounds']
        self.data_generator = DataConstraintGenerator(domain_bounds)
        
        # 生成合成数据
        if data_config.get('synthetic_data', {}).get('enabled', False):
            synthetic_config = data_config['synthetic_data']
            
            X_data, y_data = self.data_generator.generate_synthetic_data(
                num_points=synthetic_config.get('num_points', 1000),
                noise_level=synthetic_config.get('noise_level', 0.0),
                data_type=synthetic_config.get('type', 'random')
            )
            
            self.boundary_manager.add_data_constraint(
                X_data, y_data, "Synthetic data constraint"
            )
            
            print(f"Added synthetic data constraint with {len(X_data)} points")
        
        # 加载实验数据
        if 'experimental_data' in data_config:
            exp_config = data_config['experimental_data']
            file_path = exp_config.get('file_path')
            
            if file_path and os.path.exists(file_path):
                X_data, y_data = self.data_generator.load_experimental_data(file_path)
                if X_data is not None and y_data is not None:
                    self.boundary_manager.add_data_constraint(
                        X_data, y_data, "Experimental data constraint"
                    )
                    print(f"Added experimental data constraint with {len(X_data)} points")
        
        # 将数据约束添加到模型
        data_constraints = self.boundary_manager.get_data_constraints()
        for constraint in data_constraints:
            self.pinn_model.add_data_constraints(
                constraint['X_data'], constraint['y_data']
            )
    
    def setup_loss_functions(self):
        """设置损失函数"""
        print("\nSetting up loss functions...")
        
        loss_config = self.config.get('loss_functions', {})
        
        # 自适应权重调整
        if loss_config.get('adaptive_weighting', {}).get('enabled', False):
            adaptive_config = loss_config['adaptive_weighting']
            
            self.loss_weighting = AdaptiveLossWeighting(
                method=adaptive_config.get('method', 'gradient_normalization'),
                update_frequency=adaptive_config.get('update_frequency', 100),
                alpha=adaptive_config.get('alpha', 0.9)
            )
            
            print(f"Adaptive loss weighting enabled: {adaptive_config.get('method')}")
        
        # 损失监控
        self.loss_monitor = PINNLossMonitor()
        
        # 损失调度
        if 'scheduler' in loss_config:
            scheduler_config = loss_config['scheduler']
            self.loss_scheduler = LossScheduler(
                schedule_type=scheduler_config.get('type', 'exponential'),
                schedule_params=scheduler_config.get('params', {})
            )
            print(f"Loss scheduler enabled: {scheduler_config.get('type')}")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("\nSetting up optimizer...")
        
        optimizer_config = self.config.get('optimizer', {})
        
        # 两阶段优化器
        self.optimizer = TwoStageOptimizer(
            model=self.pinn_model.get_model(),
            adam_config=optimizer_config.get('adam', {}),
            lbfgs_config=optimizer_config.get('lbfgs', {})
        )
        
        # 学习率调度器
        if 'lr_scheduler' in optimizer_config:
            lr_config = optimizer_config['lr_scheduler']
            self.lr_scheduler = LearningRateScheduler(
                scheduler_type=lr_config.get('type', 'exponential'),
                scheduler_params=lr_config.get('params', {})
            )
            print(f"Learning rate scheduler enabled: {lr_config.get('type')}")
        
        # 早停机制
        if 'early_stopping' in optimizer_config:
            es_config = optimizer_config['early_stopping']
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 1000),
                min_delta=es_config.get('min_delta', 1e-6),
                restore_best_weights=es_config.get('restore_best_weights', True)
            )
            print(f"Early stopping enabled with patience: {es_config.get('patience')}")
        
        print("Optimizer setup completed")
    
    def train(self) -> Dict[str, Any]:
        """训练模型"""
        print("\n" + "="*60)
        print("Starting PINN Training")
        print("="*60)
        
        start_time = time.time()
        
        # 设置所有组件
        self.setup_model()
        self.setup_boundary_conditions()
        self.setup_data_constraints()
        self.setup_loss_functions()
        self.setup_optimizer()
        
        # 初始化损失权重
        loss_components = ['pde', 'bc', 'ic', 'data']
        if self.loss_weighting:
            initial_weights = self.loss_weighting.initialize_weights(loss_components)
        else:
            initial_weights = {comp: 1.0 for comp in loss_components}
        
        print(f"\nInitial loss weights: {initial_weights}")
        
        # 创建回调函数
        callbacks = self._create_callbacks()
        
        # 执行两阶段训练
        training_results = self.optimizer.train_full_pipeline(
            loss_weights=list(initial_weights.values()),
            callbacks=callbacks,
            save_path=os.path.join(self.save_dir, 'model')
        )
        
        # 记录训练结果
        self.training_results = training_results
        self.is_trained = True
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("Training Completed")
        print("="*60)
        print(f"Total training time: {total_time:.2f} seconds")
        
        # 保存训练历史
        self._save_training_results()
        
        return training_results
    
    def _create_callbacks(self) -> List[Callable]:
        """创建回调函数"""
        callbacks = []
        
        # 添加损失监控回调
        def loss_monitor_callback(model, train_state):
            if hasattr(train_state, 'loss_train') and len(train_state.loss_train) > 0:
                current_loss = train_state.loss_train[-1]
                iteration = len(train_state.loss_train)
                
                # 更新损失监控
                loss_values = {'total': current_loss}
                weights = self.loss_weighting.get_current_weights() if self.loss_weighting else {}
                
                self.loss_monitor.update(iteration, loss_values, weights)
        
        callbacks.append(loss_monitor_callback)
        
        return callbacks
    
    def _save_training_results(self):
        """保存训练结果"""
        # 保存配置
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.npz')
        self.optimizer.save_training_history(history_path)
        
        # 保存损失监控历史
        loss_history_path = os.path.join(self.save_dir, 'loss_history.npz')
        self.loss_monitor.save_history(loss_history_path)
        
        # 保存权重历史
        if self.loss_weighting:
            weight_plot_path = os.path.join(self.save_dir, 'weight_evolution.png')
            self.loss_weighting.plot_weight_history(weight_plot_path)
        
        # 保存训练曲线
        training_plot_path = os.path.join(self.save_dir, 'training_curves.png')
        self.optimizer.plot_training_history(training_plot_path)
        
        # 保存损失曲线
        loss_plot_path = os.path.join(self.save_dir, 'loss_curves.png')
        self.loss_monitor.plot_loss_history(loss_plot_path)
        
        print(f"\nTraining results saved to: {self.save_dir}")
    
    def evaluate(self, X_test: np.ndarray) -> Dict[str, Any]:
        """评估模型
        
        Args:
            X_test: 测试数据
        
        Returns:
            评估结果
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\nEvaluating model...")
        
        # 预测
        y_pred = self.pinn_model.predict(X_test)
        
        # 计算物理残差
        model = self.pinn_model.get_model()
        residuals = model.predict(X_test, operator=self.pinn_model.physics.pde_residual)
        
        # 评估指标
        evaluation_results = {
            'predictions': y_pred,
            'residuals': residuals,
            'residual_stats': {
                'mean': np.mean(np.abs(residuals)),
                'std': np.std(residuals),
                'max': np.max(np.abs(residuals)),
                'rms': np.sqrt(np.mean(residuals**2))
            },
            'test_points': len(X_test)
        }
        
        print(f"Evaluation completed on {len(X_test)} test points")
        print(f"Mean absolute residual: {evaluation_results['residual_stats']['mean']:.6e}")
        print(f"RMS residual: {evaluation_results['residual_stats']['rms']:.6e}")
        
        return evaluation_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 输入数据
        
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.pinn_model.predict(X)
    
    def visualize_solution(self, 
                          time_points: Optional[List[float]] = None,
                          spatial_points: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
        """可视化解
        
        Args:
            time_points: 时间点列表
            spatial_points: 空间点数组
            save_path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before visualization")
        
        print("\nVisualizing solution...")
        
        # 默认参数
        if time_points is None:
            t_bounds = self.config['model']['domain_bounds']['t']
            time_points = np.linspace(t_bounds[0], t_bounds[1], 5)
        
        if spatial_points is None:
            x_bounds = self.config['model']['domain_bounds']['x']
            spatial_points = np.linspace(x_bounds[0], x_bounds[1], 100)
        
        # 创建预测网格
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, t in enumerate(time_points[:6]):
            # 创建输入数据
            X_vis = np.zeros((len(spatial_points), 12))  # 12维输入
            X_vis[:, 0] = t  # 时间
            X_vis[:, 1] = spatial_points  # x坐标
            # 其他维度使用默认值
            X_vis[:, 4] = 1e9  # n_e
            X_vis[:, 5] = 10000  # T_e
            X_vis[:, 6] = 1e9  # n_i
            X_vis[:, 7] = 1000  # T_i
            X_vis[:, 8] = 1000  # S_flux
            
            # 预测
            V_pred = self.predict(X_vis)
            
            # 绘图
            axes[i].plot(spatial_points, V_pred.flatten(), 'b-', linewidth=2)
            axes[i].set_xlabel('Position x')
            axes[i].set_ylabel('Surface Potential V')
            axes[i].set_title(f'Time t = {t:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, 'solution_visualization.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        print(f"Solution visualization saved to: {save_path}")
    
    def save_model(self, file_path: Optional[str] = None):
        """保存模型
        
        Args:
            file_path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if file_path is None:
            file_path = os.path.join(self.save_dir, 'final_model')
        
        self.pinn_model.get_model().save(file_path)
        self.model_save_path = file_path
        
        print(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """加载模型
        
        Args:
            file_path: 模型文件路径
        """
        # 首先需要设置模型结构
        self.setup_model()
        
        # 加载权重
        self.pinn_model.get_model().restore(file_path)
        self.is_trained = True
        self.model_save_path = file_path
        
        print(f"Model loaded from: {file_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.is_trained:
            return {"status": "Not trained"}
        
        summary = {
            "status": "Trained",
            "config": self.config,
            "save_directory": self.save_dir,
            "model_save_path": self.model_save_path
        }
        
        # 添加优化器摘要
        if self.optimizer:
            summary["optimizer_summary"] = self.optimizer.get_training_summary()
        
        # 添加损失监控摘要
        if self.loss_monitor:
            summary["convergence_metrics"] = self.loss_monitor.get_convergence_metrics()
        
        return summary