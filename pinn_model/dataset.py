# 数据集处理模块
# 定义处理参数化输入的数据加载器

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import logging

class SpacecraftChargingDataset(data.Dataset):
    """
    航天器表面充电数据集
    
    处理包含以下列的CSV数据:
    - t: 时间 (s)
    - n_e: 电子密度 (m^-3)
    - T_e: 电子温度 (eV)
    - n_i: 离子密度 (m^-3)
    - T_i: 离子温度 (eV)
    - S_flux: 太阳辐射通量 (W/m^2)
    - alpha_sun: 太阳光入射角 (度)
    - material_id: 材料标识 (0: Kapton, 1: Aluminum)
    - V_ground_truth: 真实表面电位 (V) [可选，仅用于监督学习]
    """
    
    def __init__(self, data_path: str, config: Dict[str, Any], 
                 mode: str = 'train', transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            config: 配置字典
            mode: 数据集模式 ('train', 'test', 'physics')
            transform: 数据变换函数
        """
        self.data_path = Path(data_path)
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # 输入特征列名
        self.input_columns = [
            't', 'n_e', 'T_e', 'n_i', 'T_i', 
            'S_flux', 'alpha_sun', 'material_id'
        ]
        
        # 输出列名
        self.output_columns = ['V_ground_truth']
        
        # 加载数据
        self._load_data()
        
        # 计算归一化参数
        self._compute_normalization_stats()
        
        logging.info(f"Loaded {len(self)} samples from {self.data_path}")
    
    def _load_data(self):
        """
        加载CSV数据文件
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # 读取CSV文件
        self.df = pd.read_csv(self.data_path)
        
        # 验证必需的列
        missing_columns = [col for col in self.input_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 检查是否有真实值列（用于监督学习）
        self.has_ground_truth = 'V_ground_truth' in self.df.columns
        
        # 提取输入特征
        self.inputs = self.df[self.input_columns].values.astype(np.float32)
        
        # 提取输出（如果存在）
        if self.has_ground_truth:
            self.outputs = self.df[self.output_columns].values.astype(np.float32)
        else:
            self.outputs = None
        
        # 处理缺失值
        if np.isnan(self.inputs).any():
            logging.warning("Found NaN values in inputs, filling with zeros")
            self.inputs = np.nan_to_num(self.inputs, nan=0.0)
        
        if self.outputs is not None and np.isnan(self.outputs).any():
            logging.warning("Found NaN values in outputs, filling with zeros")
            self.outputs = np.nan_to_num(self.outputs, nan=0.0)
    
    def _compute_normalization_stats(self):
        """
        计算输入输出的归一化统计量
        """
        # 输入归一化统计量
        self.input_mean = torch.tensor(np.mean(self.inputs, axis=0), dtype=torch.float32)
        self.input_std = torch.tensor(np.std(self.inputs, axis=0), dtype=torch.float32)
        
        # 避免除零
        self.input_std = torch.clamp(self.input_std, min=1e-8)
        
        # 输出归一化统计量（如果存在）
        if self.outputs is not None:
            self.output_mean = torch.tensor(np.mean(self.outputs, axis=0), dtype=torch.float32)
            self.output_std = torch.tensor(np.std(self.outputs, axis=0), dtype=torch.float32)
            self.output_std = torch.clamp(self.output_std, min=1e-8)
        else:
            self.output_mean = torch.zeros(1, dtype=torch.float32)
            self.output_std = torch.ones(1, dtype=torch.float32)
    
    def get_normalization_stats(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                            Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取归一化统计量
        
        Returns:
            input_stats: (input_mean, input_std)
            output_stats: (output_mean, output_std)
        """
        return (self.input_mean, self.input_std), (self.output_mean, self.output_std)
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 包含输入和输出的字典
        """
        # 获取输入
        inputs = torch.tensor(self.inputs[idx], dtype=torch.float32)
        
        sample = {'inputs': inputs}
        
        # 获取输出（如果存在）
        if self.outputs is not None:
            outputs = torch.tensor(self.outputs[idx], dtype=torch.float32)
            sample['outputs'] = outputs
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        获取批量数据
        
        Args:
            indices: 样本索引列表
            
        Returns:
            batch: 批量数据字典
        """
        batch_inputs = torch.stack([self[i]['inputs'] for i in indices])
        batch = {'inputs': batch_inputs}
        
        if self.outputs is not None:
            batch_outputs = torch.stack([self[i]['outputs'] for i in indices])
            batch['outputs'] = batch_outputs
        
        return batch
    
    def get_scenario_data(self, scenario_id: int) -> Dict[str, torch.Tensor]:
        """
        获取特定场景的所有时间步数据
        
        Args:
            scenario_id: 场景ID（假设数据按场景分组）
            
        Returns:
            scenario_data: 场景数据
        """
        # 这里假设数据中有scenario_id列，如果没有则返回所有数据
        if 'scenario_id' in self.df.columns:
            mask = self.df['scenario_id'] == scenario_id
            scenario_indices = self.df.index[mask].tolist()
        else:
            # 如果没有scenario_id，返回所有数据
            scenario_indices = list(range(len(self)))
        
        return self.get_batch(scenario_indices)

class PhysicsDataset(data.Dataset):
    """
    物理约束数据集
    
    用于生成满足物理约束的训练点，不需要真实标签
    """
    
    def __init__(self, config: Dict[str, Any], num_samples: int = 10000):
        """
        初始化物理数据集
        
        Args:
            config: 配置字典
            num_samples: 样本数量
        """
        self.config = config
        self.num_samples = num_samples
        
        # 获取参数范围
        self.data_config = config['data_generation']
        
        # 生成随机样本
        self._generate_samples()
    
    def _generate_samples(self):
        """
        生成随机物理约束样本
        """
        samples = []
        
        for _ in range(self.num_samples):
            # 随机采样时间
            t = np.random.uniform(
                self.data_config['t_start'], 
                self.data_config['t_end']
            )
            
            # 随机采样环境参数
            n_e = np.random.uniform(float(self.data_config['n_e_range'][0]), 
                                    float(self.data_config['n_e_range'][1]))
            T_e = np.random.uniform(float(self.data_config['T_e_range'][0]), 
                                    float(self.data_config['T_e_range'][1]))
            n_i = np.random.uniform(float(self.data_config['n_i_range'][0]), 
                                    float(self.data_config['n_i_range'][1]))
            T_i = np.random.uniform(float(self.data_config['T_i_range'][0]), 
                                    float(self.data_config['T_i_range'][1]))
            
            # 随机采样光照参数
            S_flux = np.random.uniform(float(self.data_config['S_flux_range'][0]), 
                                    float(self.data_config['S_flux_range'][1]))
            alpha_sun = np.random.uniform(float(self.data_config['alpha_sun_range'][0]), 
                                    float(self.data_config['alpha_sun_range'][1]))
            
            # 随机选择材料
            material_id = np.random.choice(self.data_config['materials'])
            
            sample = [t, n_e, T_e, n_i, T_i, S_flux, alpha_sun, material_id]
            samples.append(sample)
        
        self.samples = np.array(samples, dtype=np.float32)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = torch.tensor(self.samples[idx], dtype=torch.float32)
        return {'inputs': inputs}

class DataLoader:
    """
    数据加载器管理类
    
    统一管理训练、测试和物理约束数据集
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.training_config = config['training']
        
        # 数据集实例
        self.train_dataset = None
        self.test_dataset = None
        self.physics_dataset = None
        
        # 数据加载器实例
        self.train_loader = None
        self.test_loader = None
        self.physics_loader = None
    
    def load_datasets(self, train_path: str, test_path: Optional[str] = None,
                     physics_samples: int = 10000):
        """
        加载所有数据集
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径（可选）
            physics_samples: 物理约束样本数量
        """
        # 加载训练数据集
        self.train_dataset = SpacecraftChargingDataset(
            train_path, self.config, mode='train'
        )
        
        # 加载测试数据集（如果提供）
        if test_path:
            self.test_dataset = SpacecraftChargingDataset(
                test_path, self.config, mode='test'
            )
        
        # 创建物理约束数据集
        self.physics_dataset = PhysicsDataset(self.config, physics_samples)
        
        logging.info(f"Loaded datasets: train={len(self.train_dataset)}, "
                    f"test={len(self.test_dataset) if self.test_dataset else 0}, "
                    f"physics={len(self.physics_dataset)}")
    
    def create_data_loaders(self, batch_size: Optional[int] = None):
        """
        创建PyTorch数据加载器
        
        Args:
            batch_size: 批大小（如果不提供则使用配置中的值）
        """
        if batch_size is None:
            batch_size = self.training_config['adam_phase']['batch_size']
        
        # 训练数据加载器
        if self.train_dataset:
            self.train_loader = data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Windows兼容性
                pin_memory=True
            )
        
        # 测试数据加载器
        if self.test_dataset:
            self.test_loader = data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # 物理约束数据加载器
        if self.physics_dataset:
            self.physics_loader = data.DataLoader(
                self.physics_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
    
    def get_normalization_stats(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                            Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取归一化统计量（从训练数据集）
        
        Returns:
            input_stats: (input_mean, input_std)
            output_stats: (output_mean, output_std)
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset not loaded")
        
        return self.train_dataset.get_normalization_stats()

def create_data_loader(config: Dict[str, Any]) -> DataLoader:
    """
    创建数据加载器实例
    
    Args:
        config: 配置字典
        
    Returns:
        data_loader: 数据加载器实例
    """
    return DataLoader(config)

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义批处理函数
    
    Args:
        batch: 批量样本列表
        
    Returns:
        batched_data: 批处理后的数据
    """
    # 提取输入
    inputs = torch.stack([sample['inputs'] for sample in batch])
    result = {'inputs': inputs}
    
    # 提取输出（如果存在）
    if 'outputs' in batch[0]:
        outputs = torch.stack([sample['outputs'] for sample in batch])
        result['outputs'] = outputs
    
    return result

def validate_data_format(data_path: str) -> bool:
    """
    验证数据文件格式
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        is_valid: 是否有效
    """
    try:
        df = pd.read_csv(data_path)
        
        required_columns = [
            't', 'n_e', 'T_e', 'n_i', 'T_i', 
            'S_flux', 'alpha_sun', 'material_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
        
        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.error(f"Column {col} is not numeric")
                return False
        
        # 检查数据范围
        if (df['material_id'] < 0).any() or (df['material_id'] > 1).any():
            logging.error("material_id should be 0 or 1")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating data format: {e}")
        return False