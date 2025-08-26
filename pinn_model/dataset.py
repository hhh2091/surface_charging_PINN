"""数据集处理模块

处理航天器表面充电PINN的参数化输入数据，包括：
1. 数据加载和预处理
2. 参数化输入处理
3. 训练/测试数据集分割
4. 批处理和数据增强

Author: PINN Engineering Team
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


class SpacecraftChargingDataset(Dataset):
    """航天器表面充电数据集类
    
    处理包含以下列的CSV数据：
    [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id, V_ground_truth]
    """
    
    def __init__(self, data_path: str, config: Dict, 
                 mode: str = 'train', transform: Optional[callable] = None):
        """初始化数据集
        
        Args:
            data_path: 数据文件路径
            config: 配置字典
            mode: 模式 ('train', 'test', 'val')
            transform: 数据变换函数
        """
        self.data_path = data_path
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # 加载数据
        self.data = self._load_data()
        
        # 数据预处理
        self.processed_data = self._preprocess_data()
        
        # 输入输出分离
        self.inputs, self.targets = self._split_inputs_targets()
        
        print(f"Loaded {mode} dataset: {len(self)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据文件
        
        Returns:
            data: 加载的数据DataFrame
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # 支持多种文件格式
        if self.data_path.endswith('.csv'):
            data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl'):
            data = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        # 验证数据列
        expected_columns = ['t', 'n_e', 'T_e', 'n_i', 'T_i', 'S_flux', 
                          'alpha_sun', 'material_id', 'V_ground_truth']
        
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        return data
    
    def _preprocess_data(self) -> pd.DataFrame:
        """数据预处理
        
        Returns:
            processed_data: 预处理后的数据
        """
        data = self.data.copy()
        
        # 处理缺失值
        if data.isnull().any().any():
            print("Warning: Found missing values, filling with forward fill")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 数据类型转换
        data['material_id'] = data['material_id'].astype(int)
        
        # 角度转换（度到弧度，如果需要）
        # data['alpha_sun'] = np.deg2rad(data['alpha_sun'])
        
        # 数据范围检查
        self._validate_data_ranges(data)
        
        return data
    
    def _validate_data_ranges(self, data: pd.DataFrame):
        """验证数据范围
        
        Args:
            data: 待验证的数据
        """
        # 检查物理量的合理范围
        checks = {
            'n_e': (1e4, 1e10),  # 电子密度 (m^-3)
            'T_e': (0.01, 100),  # 电子温度 (eV)
            'n_i': (1e4, 1e10),  # 离子密度 (m^-3)
            'T_i': (0.01, 100),  # 离子温度 (eV)
            'S_flux': (0, 2000), # 太阳通量 (W/m^2)
            'alpha_sun': (0, 90), # 太阳入射角 (degrees)
            't': (0, None),      # 时间 (s)
        }
        
        for col, (min_val, max_val) in checks.items():
            if col in data.columns:
                col_data = data[col]
                if min_val is not None and (col_data < min_val).any():
                    print(f"Warning: {col} has values below {min_val}")
                if max_val is not None and (col_data > max_val).any():
                    print(f"Warning: {col} has values above {max_val}")
    
    def _split_inputs_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """分离输入和目标
        
        Returns:
            inputs: 输入张量 [N, 8]
            targets: 目标张量 [N, 1]
        """
        # 输入特征：[t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
        input_columns = ['t', 'n_e', 'T_e', 'n_i', 'T_i', 'S_flux', 'alpha_sun', 'material_id']
        target_columns = ['V_ground_truth']
        
        inputs = torch.tensor(self.processed_data[input_columns].values, dtype=torch.float32)
        targets = torch.tensor(self.processed_data[target_columns].values, dtype=torch.float32)
        
        return inputs, targets
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            input_sample: 输入样本 [8]
            target_sample: 目标样本 [1]
        """
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]
        
        if self.transform:
            input_sample = self.transform(input_sample)
        
        return input_sample, target_sample
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'size': len(self),
            'input_shape': self.inputs.shape,
            'target_shape': self.targets.shape,
            'input_mean': self.inputs.mean(dim=0).tolist(),
            'input_std': self.inputs.std(dim=0).tolist(),
            'target_mean': self.targets.mean().item(),
            'target_std': self.targets.std().item(),
            'material_distribution': self.processed_data['material_id'].value_counts().to_dict()
        }


class DataScaler:
    """数据缩放器
    
    用于输入特征的标准化和归一化。
    """
    
    def __init__(self, method: str = 'standard'):
        """初始化缩放器
        
        Args:
            method: 缩放方法 ('standard', 'minmax', 'log_standard')
        """
        self.method = method
        self.scaler = None
        self.fitted = False
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'log_standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    def fit(self, data: torch.Tensor) -> 'DataScaler':
        """拟合缩放器
        
        Args:
            data: 训练数据 [N, features]
            
        Returns:
            self: 返回自身以支持链式调用
        """
        data_np = data.numpy()
        
        if self.method == 'log_standard':
            # 对某些特征应用对数变换（如密度）
            log_features = [1, 3]  # n_e, n_i的索引
            data_np[:, log_features] = np.log10(data_np[:, log_features] + 1e-10)
        
        self.scaler.fit(data_np)
        self.fitted = True
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """应用缩放变换
        
        Args:
            data: 待变换数据 [N, features]
            
        Returns:
            scaled_data: 缩放后的数据
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        data_np = data.numpy()
        
        if self.method == 'log_standard':
            log_features = [1, 3]  # n_e, n_i的索引
            data_np[:, log_features] = np.log10(data_np[:, log_features] + 1e-10)
        
        scaled_data = self.scaler.transform(data_np)
        return torch.tensor(scaled_data, dtype=torch.float32)
    
    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        """逆变换
        
        Args:
            scaled_data: 缩放后的数据
            
        Returns:
            original_data: 原始尺度的数据
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        data_np = self.scaler.inverse_transform(scaled_data.numpy())
        
        if self.method == 'log_standard':
            log_features = [1, 3]  # n_e, n_i的索引
            data_np[:, log_features] = 10 ** data_np[:, log_features]
        
        return torch.tensor(data_np, dtype=torch.float32)


class PINNDataLoader:
    """PINN数据加载器
    
    专门为PINN训练设计的数据加载器，支持物理约束点和数据点的混合采样。
    """
    
    def __init__(self, config: Dict):
        """初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.batch_size = config['training']['batch_size']
        
        # 数据缩放器
        self.input_scaler = DataScaler(method='log_standard')
        self.target_scaler = DataScaler(method='standard')
        
        # 数据集
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
    
    def load_datasets(self, train_path: str, test_path: Optional[str] = None, 
                     val_path: Optional[str] = None):
        """加载数据集
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径（可选）
            val_path: 验证数据路径（可选）
        """
        # 加载训练数据集
        self.train_dataset = SpacecraftChargingDataset(
            train_path, self.config, mode='train'
        )
        
        # 拟合缩放器
        self.input_scaler.fit(self.train_dataset.inputs)
        self.target_scaler.fit(self.train_dataset.targets)
        
        # 加载测试数据集
        if test_path:
            self.test_dataset = SpacecraftChargingDataset(
                test_path, self.config, mode='test'
            )
        
        # 加载验证数据集
        if val_path:
            self.val_dataset = SpacecraftChargingDataset(
                val_path, self.config, mode='val'
            )
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """获取PyTorch数据加载器
        
        Returns:
            dataloaders: 数据加载器字典
        """
        dataloaders = {}
        
        if self.train_dataset:
            dataloaders['train'] = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Windows兼容性
                pin_memory=True
            )
        
        if self.test_dataset:
            dataloaders['test'] = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        if self.val_dataset:
            dataloaders['val'] = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        return dataloaders
    
    def generate_physics_points(self, n_points: int) -> torch.Tensor:
        """生成物理约束点
        
        Args:
            n_points: 点的数量
            
        Returns:
            physics_points: 物理约束点 [n_points, 8]
        """
        # 从配置中获取参数范围
        data_config = self.config['data_generation']
        
        # 生成随机采样点
        t = torch.rand(n_points, 1) * data_config['t_end']
        
        n_e_min, n_e_max = data_config['n_e_range']
        n_e = torch.rand(n_points, 1) * (n_e_max - n_e_min) + n_e_min
        
        T_e_min, T_e_max = data_config['T_e_range']
        T_e = torch.rand(n_points, 1) * (T_e_max - T_e_min) + T_e_min
        
        n_i_min, n_i_max = data_config['n_i_range']
        n_i = torch.rand(n_points, 1) * (n_i_max - n_i_min) + n_i_min
        
        T_i_min, T_i_max = data_config['T_i_range']
        T_i = torch.rand(n_points, 1) * (T_i_max - T_i_min) + T_i_min
        
        S_flux_min, S_flux_max = data_config['S_flux_range']
        S_flux = torch.rand(n_points, 1) * (S_flux_max - S_flux_min) + S_flux_min
        
        alpha_min, alpha_max = data_config['alpha_sun_range']
        alpha_sun = torch.rand(n_points, 1) * (alpha_max - alpha_min) + alpha_min
        
        # 随机选择材料
        materials = data_config['materials']
        material_id = torch.randint(0, len(materials), (n_points, 1)).float()
        
        # 组合所有特征
        physics_points = torch.cat([
            t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id
        ], dim=1)
        
        return physics_points
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息
        
        Returns:
            info: 数据集信息字典
        """
        info = {'datasets': {}}
        
        if self.train_dataset:
            info['datasets']['train'] = self.train_dataset.get_statistics()
        
        if self.test_dataset:
            info['datasets']['test'] = self.test_dataset.get_statistics()
        
        if self.val_dataset:
            info['datasets']['val'] = self.val_dataset.get_statistics()
        
        return info


def create_dataloader(config: Dict) -> PINNDataLoader:
    """创建数据加载器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        dataloader: PINN数据加载器
    """
    return PINNDataLoader(config)