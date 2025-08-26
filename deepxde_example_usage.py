import numpy as np
import torch
import os
import json
from typing import Dict, Any

# 导入自定义模块
from deepxde_trainer import DeepXDEPINNTrainer

def create_config() -> Dict[str, Any]:
    """创建训练配置"""
    config = {
        # 模型配置
        'model': {
            'domain_bounds': {
                't': [0.0, 1.0],      # 时间范围
                'x': [0.0, 1.0],      # x坐标范围
                'y': [0.0, 1.0],      # y坐标范围
                'z': [0.0, 1.0],      # z坐标范围
                'n_e': [1e8, 1e12],   # 电子密度范围
                'T_e': [1000, 50000], # 电子温度范围
                'n_i': [1e8, 1e12],   # 离子密度范围
                'T_i': [300, 10000],  # 离子温度范围
                'S_flux': [100, 2000], # 太阳辐射通量范围
                'alpha_sun': [0, np.pi/2], # 太阳入射角范围
                'alpha_ram': [0, np.pi],   # 迎风角范围
                'material_id': [0, 3]      # 材料ID范围
            },
            'physical_params': {
                'C': 1e-9,           # 单位面积电容 (F/m²)
                'q_e': -1.602e-19,   # 电子电荷 (C)
                'q_i': 1.602e-19,    # 离子电荷 (C)
                'k_B': 1.381e-23,    # 玻尔兹曼常数 (J/K)
                'm_e': 9.109e-31,    # 电子质量 (kg)
                'm_i': 1.673e-27,    # 离子质量 (kg)
                'J_ph0': 1e-6,       # 饱和光电子电流密度 (A/m²)
                'V_ph': 2.0,         # 等效光电子温度 (V)
                'material_properties': {
                    0: {'work_function': 4.5, 'secondary_yield': 1.2},  # 铝
                    1: {'work_function': 5.1, 'secondary_yield': 0.8},  # 钛
                    2: {'work_function': 4.3, 'secondary_yield': 1.5},  # 银
                    3: {'work_function': 4.8, 'secondary_yield': 1.0}   # 不锈钢
                }
            },
            'network_config': {
                'input_dim': 12,
                'output_dim': 1,
                'hidden_layers': [64, 64, 64, 64],
                'activation': 'tanh',
                'initialization': 'Glorot normal',
                'yield_network': {
                    'hidden_layers': [32, 32, 32],
                    'activation': 'tanh',
                    'initialization': 'Glorot normal'
                }
            }
        },
        
        # 边界条件配置
        'boundary_conditions': {
            'grounded_surface': False,
            'floating_potential': -5.0,  # 浮动电位 (V)
            'initial_conditions': {
                'type': 'zero',  # 'zero', 'constant', 'gaussian'
                'value': 0.0,
                'center': [0.5, 0.5, 0.5],
                'amplitude': 1.0,
                'width': 0.1
            }
        },
        
        # 数据约束配置
        'data_constraints': {
            'enabled': True,
            'synthetic_data': {
                'enabled': True,
                'num_points': 1000,
                'noise_level': 0.01,
                'type': 'random'  # 'random', 'grid', 'boundary'
            },
            'experimental_data': {
                'file_path': './data/experimental_data.csv'  # 可选
            }
        },
        
        # 损失函数配置
        'loss_functions': {
            'adaptive_weighting': {
                'enabled': True,
                'method': 'gradient_normalization',  # 'gradient_normalization', 'loss_balancing', 'annealing'
                'update_frequency': 100,
                'alpha': 0.9
            },
            'scheduler': {
                'type': 'exponential',  # 'exponential', 'linear', 'cosine'
                'params': {
                    'decay_rate': 0.95,
                    'decay_steps': 1000
                }
            }
        },
        
        # 优化器配置
        'optimizer': {
            'adam': {
                'epochs': 10000,
                'learning_rate': 1e-3,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8
            },
            'lbfgs': {
                'epochs': 5000,
                'learning_rate': 1.0,
                'max_iter': 50000,
                'max_eval': None,
                'tolerance_grad': 1e-7,
                'tolerance_change': 1e-9,
                'history_size': 100
            },
            'lr_scheduler': {
                'type': 'exponential',  # 'exponential', 'step', 'cosine'
                'params': {
                    'gamma': 0.95,
                    'step_size': 1000
                }
            },
            'early_stopping': {
                'patience': 2000,
                'min_delta': 1e-6,
                'restore_best_weights': True
            }
        },
        
        # 训练配置
        'training': {
            'num_domain_points': 10000,
            'num_boundary_points': 2000,
            'num_initial_points': 1000,
            'batch_size': None,  # None表示使用全部数据
            'validation_split': 0.1,
            'shuffle': True,
            'seed': 42
        },
        
        # 保存配置
        'save_dir': './results/deepxde_pinn_experiment',
        'save_frequency': 1000,
        'checkpoint_frequency': 5000
    }
    
    return config

def main():
    """主函数"""
    print("DeepXDE PINN Surface Charging Example")
    print("="*50)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建配置
    config = create_config()
    
    # 保存配置
    os.makedirs(config['save_dir'], exist_ok=True)
    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # 创建训练器
    trainer = DeepXDEPINNTrainer(config)
    
    # 训练模型
    print("\nStarting training...")
    training_results = trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    # 生成测试数据
    print("\nGenerating test data...")
    domain_bounds = config['model']['domain_bounds']
    
    # 创建测试网格
    t_test = np.linspace(domain_bounds['t'][0], domain_bounds['t'][1], 10)
    x_test = np.linspace(domain_bounds['x'][0], domain_bounds['x'][1], 50)
    
    X_test = []
    for t in t_test:
        for x in x_test:
            # 创建4维输入向量 (t, x, y, z)
            input_vec = np.array([
                t,           # 时间
                x,           # x坐标
                0.5,         # y坐标
                0.5          # z坐标
            ])
            X_test.append(input_vec)
    
    X_test = np.array(X_test)
    
    # 评估模型
    print("\nEvaluating model...")
    evaluation_results = trainer.evaluate(X_test)
    
    # 可视化解
    print("\nVisualizing solution...")
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    trainer.visualize_solution(
        time_points=time_points,
        save_path=os.path.join(config['save_dir'], 'solution_evolution.png')
    )
    
    # 打印训练摘要
    print("\nTraining Summary:")
    print("="*50)
    summary = trainer.get_training_summary()
    
    if 'optimizer_summary' in summary:
        opt_summary = summary['optimizer_summary']
        print(f"Adam stage - Final loss: {opt_summary.get('adam_final_loss', 'N/A'):.6e}")
        print(f"L-BFGS stage - Final loss: {opt_summary.get('lbfgs_final_loss', 'N/A'):.6e}")
        print(f"Total training time: {opt_summary.get('total_time', 'N/A'):.2f} seconds")
    
    if 'convergence_metrics' in summary:
        conv_metrics = summary['convergence_metrics']
        print(f"Convergence achieved: {conv_metrics.get('converged', 'Unknown')}")
        print(f"Final loss: {conv_metrics.get('final_loss', 'N/A'):.6e}")
    
    print(f"\nResults saved to: {config['save_dir']}")
    
    return trainer, evaluation_results

def demo_prediction(model_path: str, config_path: str):
    """演示预测功能
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
    """
    print("\nDemo: Loading trained model and making predictions")
    print("="*50)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建训练器并加载模型
    trainer = DeepXDEPINNTrainer(config)
    trainer.load_model(model_path)
    
    # 创建预测点
    domain_bounds = config['model']['domain_bounds']
    
    # 时间演化预测
    t_pred = np.linspace(0, 1, 100)
    x_fixed = 0.5  # 固定位置
    
    X_pred = []
    for t in t_pred:
        input_vec = np.array([
            t,           # 时间
            x_fixed,     # x坐标
            0.5,         # y坐标
            0.5,         # z坐标
            5e9,         # 电子密度
            15000,       # 电子温度
            5e9,         # 离子密度
            1500,        # 离子温度
            800,         # 太阳辐射通量
            np.pi/6,     # 太阳入射角
            np.pi/3,     # 迎风角
            1            # 材料ID (钛)
        ])
        X_pred.append(input_vec)
    
    X_pred = np.array(X_pred)
    
    # 进行预测
    V_pred = trainer.predict(X_pred)
    
    # 绘制时间演化
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_pred, V_pred.flatten(), 'b-', linewidth=2, label='Surface Potential')
    plt.xlabel('Time')
    plt.ylabel('Surface Potential (V)')
    plt.title(f'Surface Potential Evolution at x = {x_fixed}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(os.path.dirname(model_path), 'prediction_demo.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Prediction demo completed")
    print(f"Plot saved to: {save_path}")
    print(f"Final potential: {V_pred[-1][0]:.4f} V")
    print(f"Potential range: [{V_pred.min():.4f}, {V_pred.max():.4f}] V")

if __name__ == "__main__":
    # 运行主训练流程
    trainer, results = main()
    
    # 演示预测功能
    if trainer.model_save_path:
        config_path = os.path.join(trainer.save_dir, 'config.json')
        demo_prediction(trainer.model_save_path, config_path)
    
    print("\nExample completed successfully!")