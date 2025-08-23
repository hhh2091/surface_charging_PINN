#!/usr/bin/env python3
"""
表面充电PINN框架使用示例

这个示例展示了如何使用PINN框架来解决表面充电问题。
包括模型配置、训练、评估和可视化。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from trainer import PINNTrainer
from data_generator import DataGenerator
import json
import os

def create_config():
    """创建训练配置"""
    config = {
        # 模型配置
        'model': {
            'vnn_input_dim': 12,  # [t, x, y, z, ne, Te, ni, Ti, Sflux, alpha_sun, alpha_ram, material_id]
            'vnn_hidden_dims': [64, 64, 64, 64],
            'vnn_activation': 'tanh',
            'yield_input_dim': 4,  # [V, Je_inc, Mprop]
            'yield_hidden_dims': [32, 32, 32],
            'yield_activation': 'relu'
        },
        
        # 数据配置
        'data': {
            # 时空域范围
            't_range': [0.0, 10.0],  # 时间范围 (s)
            'x_range': [-0.1, 0.1],  # 空间x范围 (m)
            'y_range': [-0.1, 0.1],  # 空间y范围 (m)
            'z_range': [-0.01, 0.01],  # 空间z范围 (m)
            
            # 物理参数范围
            'ne_range': [1e6, 1e8],    # 电子密度范围 (m^-3)
            'Te_range': [1000, 5000],  # 电子温度范围 (K)
            'ni_range': [1e6, 1e8],    # 离子密度范围 (m^-3)
            'Ti_range': [1000, 5000],  # 离子温度范围 (K)
            
            # 光照参数范围
            'Sflux_range': [0.0, 1361.0],  # 太阳辐射通量范围 (W/m^2)
            'alpha_sun_range': [0.0, np.pi/2],  # 太阳入射角范围
            'alpha_ram_range': [0.0, np.pi],    # 迎风角范围
            
            # 材料类型
            'n_materials': 3,
            
            # 数据点数量
            'n_collocation': 10000,  # 配置点数量
            'n_boundary': 1000,      # 边界点数量
            'n_initial': 1000,       # 初始点数量
            'n_data': 2000,          # 数据点数量
            'noise_level': 0.01,     # 数据噪声水平
            'sampling_method': 'latin_hypercube'  # 采样方法
        },
        
        # 损失函数权重
        'loss_weights': {
            'residual': 1.0,
            'boundary': 10.0,
            'initial': 10.0,
            'data': 1.0
        },
        
        # 自适应权重调整
        'adaptive_weighting': {
            'method': 'gradient_normalization',
            'alpha': 0.9
        },
        'use_adaptive_weighting': True,
        
        # 优化器配置
        'optimizer': {
            'adam': {
                'lr': 1e-3,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 1e-6,
                'max_iter': 5000
            },
            'lbfgs': {
                'lr': 1.0,
                'max_iter': 1000,
                'tolerance_grad': 1e-7,
                'tolerance_change': 1e-9,
                'history_size': 100,
                'line_search_fn': 'strong_wolfe'
            }
        },
        
        # 数据归一化
        'normalize_inputs': True,
        'normalize_outputs': True,
        'input_normalization': 'minmax',
        'output_normalization': 'zscore',
        
        # 早停机制
        'early_stopping': {
            'patience': 200,
            'min_delta': 1e-6,
            'restore_best_weights': True
        },
        
        # 梯度裁剪
        'gradient_clipping': True,
        'max_grad_norm': 1.0,
        
        # 保存路径
        'save_dir': './results'
    }
    
    return config

def main():
    """主函数"""
    print("=" * 60)
    print("表面充电PINN框架示例")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置
    config = create_config()
    
    # 保存配置
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器
    print("\n创建PINN训练器...")
    trainer = PINNTrainer(config, device)
    
    # 训练模型
    print("\n开始训练...")
    training_results = trainer.train(verbose=True)
    
    # 保存模型
    model_path = os.path.join(config['save_dir'], 'pinn_model.pth')
    trainer.save_model(model_path)
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    history_path = os.path.join(config['save_dir'], 'training_history.png')
    trainer.plot_training_history(save_path=history_path)
    
    # 生成测试数据
    print("\n生成测试数据...")
    test_generator = DataGenerator(config['data'], device)
    test_points, test_values = test_generator.generate_synthetic_data(1000, noise_level=0.0)
    
    # 评估模型
    print("\n评估模型性能...")
    metrics = trainer.evaluate(test_points, test_values)
    
    print("\n评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6e}")
    
    # 可视化解
    print("\n可视化解...")
    solution_path = os.path.join(config['save_dir'], 'solution_visualization.png')
    trainer.plot_solution(t_slice=5.0, save_path=solution_path)
    
    # 保存训练结果
    results_path = os.path.join(config['save_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                           for k, v in value.items()}
            elif isinstance(value, (list, np.ndarray)):
                serializable_results[key] = [float(x) if isinstance(x, (int, float, np.number)) else x 
                                           for x in value]
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n训练完成！结果保存在: {config['save_dir']}")
    print("\n文件列表:")
    for file in os.listdir(config['save_dir']):
        print(f"  - {file}")

def demo_prediction():
    """演示预测功能"""
    print("\n=" * 60)
    print("预测演示")
    print("=" * 60)
    
    # 加载配置和模型
    config = create_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = PINNTrainer(config, device)
    
    # 尝试加载已训练的模型
    model_path = os.path.join(config['save_dir'], 'pinn_model.pth')
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print("未找到已训练的模型，请先运行训练")
        return
    
    # 创建预测点
    print("\n创建预测点...")
    n_pred = 100
    t_vals = np.linspace(0, 10, n_pred)
    
    # 固定其他参数
    pred_points = []
    for t in t_vals:
        point = [t, 0.0, 0.0, 0.0,  # t, x, y, z (中心点)
                1e7, 3000, 1e7, 3000,  # ne, Te, ni, Ti (典型等离子体参数)
                1361, 0.0, 0.0, 0]     # Sflux, alpha_sun, alpha_ram, material_id
        pred_points.append(point)
    
    pred_points = torch.tensor(pred_points, dtype=torch.float32, device=device)
    
    # 进行预测
    print("\n进行预测...")
    predictions = trainer.predict(pred_points)
    predictions = predictions.cpu().numpy().flatten()
    
    # 绘制时间演化
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, predictions, 'b-', linewidth=2, label='PINN预测')
    plt.xlabel('时间 (s)')
    plt.ylabel('表面电位 (V)')
    plt.title('表面电位时间演化 (中心点)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图像
    pred_path = os.path.join(config['save_dir'], 'time_evolution.png')
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n时间演化图保存至: {pred_path}")
    print(f"预测范围: [{predictions.min():.6f}, {predictions.max():.6f}] V")

if __name__ == '__main__':
    # 运行主训练示例
    main()
    
    # 运行预测演示
    demo_prediction()
    
    print("\n=" * 60)
    print("示例运行完成！")
    print("=" * 60)