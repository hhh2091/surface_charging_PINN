import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class DataGenerator:
    """数据生成器，用于生成PINN训练所需的各种数据点"""
    
    def __init__(self, config: Dict, device: torch.device = torch.device('cpu')):
        """
        Args:
            config: 数据生成配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 时空域范围
        self.t_range = config.get('t_range', [0.0, 1.0])  # 时间范围
        self.x_range = config.get('x_range', [-1.0, 1.0])  # 空间x范围
        self.y_range = config.get('y_range', [-1.0, 1.0])  # 空间y范围
        self.z_range = config.get('z_range', [-1.0, 1.0])  # 空间z范围
        
        # 物理参数范围
        self.ne_range = config.get('ne_range', [1e6, 1e8])    # 电子密度范围 (m^-3)
        self.Te_range = config.get('Te_range', [1000, 5000])  # 电子温度范围 (K)
        self.ni_range = config.get('ni_range', [1e6, 1e8])    # 离子密度范围 (m^-3)
        self.Ti_range = config.get('Ti_range', [1000, 5000])  # 离子温度范围 (K)
        
        # 光照参数范围
        self.Sflux_range = config.get('Sflux_range', [0.0, 1361.0])  # 太阳辐射通量范围 (W/m^2)
        self.alpha_sun_range = config.get('alpha_sun_range', [0.0, np.pi/2])  # 太阳入射角范围
        self.alpha_ram_range = config.get('alpha_ram_range', [0.0, np.pi])    # 迎风角范围
        
        # 材料类型数量
        self.n_materials = config.get('n_materials', 3)
        
    def generate_collocation_points(self, n_points: int, 
                                  sampling_method: str = 'uniform') -> torch.Tensor:
        """生成配置点（用于物理残差计算）
        
        Args:
            n_points: 点的数量
            sampling_method: 采样方法 ('uniform', 'latin_hypercube', 'sobol')
        
        Returns:
            collocation_points: 配置点 [n_points, input_dim]
        """
        if sampling_method == 'uniform':
            points = self._uniform_sampling(n_points)
        elif sampling_method == 'latin_hypercube':
            points = self._latin_hypercube_sampling(n_points)
        elif sampling_method == 'sobol':
            points = self._sobol_sampling(n_points)
        else:
            points = self._uniform_sampling(n_points)
        
        return torch.tensor(points, dtype=torch.float32, device=self.device)
    
    def _uniform_sampling(self, n_points: int) -> np.ndarray:
        """均匀采样"""
        points = np.random.rand(n_points, 12)  # 12维输入
        
        # 时空坐标
        points[:, 0] = points[:, 0] * (self.t_range[1] - self.t_range[0]) + self.t_range[0]  # t
        points[:, 1] = points[:, 1] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]  # x
        points[:, 2] = points[:, 2] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]  # y
        points[:, 3] = points[:, 3] * (self.z_range[1] - self.z_range[0]) + self.z_range[0]  # z
        
        # 环境参数
        points[:, 4] = points[:, 4] * (self.ne_range[1] - self.ne_range[0]) + self.ne_range[0]  # ne
        points[:, 5] = points[:, 5] * (self.Te_range[1] - self.Te_range[0]) + self.Te_range[0]  # Te
        points[:, 6] = points[:, 6] * (self.ni_range[1] - self.ni_range[0]) + self.ni_range[0]  # ni
        points[:, 7] = points[:, 7] * (self.Ti_range[1] - self.Ti_range[0]) + self.Ti_range[0]  # Ti
        
        # 光照参数
        points[:, 8] = points[:, 8] * (self.Sflux_range[1] - self.Sflux_range[0]) + self.Sflux_range[0]  # Sflux
        points[:, 9] = points[:, 9] * (self.alpha_sun_range[1] - self.alpha_sun_range[0]) + self.alpha_sun_range[0]  # alpha_sun
        points[:, 10] = points[:, 10] * (self.alpha_ram_range[1] - self.alpha_ram_range[0]) + self.alpha_ram_range[0]  # alpha_ram
        
        # 材料标识（整数）
        points[:, 11] = np.random.randint(0, self.n_materials, n_points).astype(float)  # material_id
        
        return points
    
    def _latin_hypercube_sampling(self, n_points: int) -> np.ndarray:
        """拉丁超立方采样"""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=12)
            points = sampler.random(n=n_points)
        except ImportError:
            # 如果没有scipy.stats.qmc，使用简单的分层采样
            points = np.random.rand(n_points, 12)
            for i in range(12):
                points[:, i] = (np.random.permutation(n_points) + points[:, i]) / n_points
        
        # 缩放到实际范围
        return self._scale_to_ranges(points)
    
    def _sobol_sampling(self, n_points: int) -> np.ndarray:
        """Sobol序列采样"""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=12, scramble=True)
            points = sampler.random(n=n_points)
        except ImportError:
            # 如果没有scipy.stats.qmc，回退到均匀采样
            points = np.random.rand(n_points, 12)
        
        return self._scale_to_ranges(points)
    
    def _scale_to_ranges(self, points: np.ndarray) -> np.ndarray:
        """将[0,1]范围的点缩放到实际物理范围"""
        scaled_points = points.copy()
        
        # 时空坐标
        scaled_points[:, 0] = points[:, 0] * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        scaled_points[:, 1] = points[:, 1] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        scaled_points[:, 2] = points[:, 2] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        scaled_points[:, 3] = points[:, 3] * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        
        # 环境参数（对数缩放）
        scaled_points[:, 4] = np.exp(points[:, 4] * (np.log(self.ne_range[1]) - np.log(self.ne_range[0])) + np.log(self.ne_range[0]))
        scaled_points[:, 5] = points[:, 5] * (self.Te_range[1] - self.Te_range[0]) + self.Te_range[0]
        scaled_points[:, 6] = np.exp(points[:, 6] * (np.log(self.ni_range[1]) - np.log(self.ni_range[0])) + np.log(self.ni_range[0]))
        scaled_points[:, 7] = points[:, 7] * (self.Ti_range[1] - self.Ti_range[0]) + self.Ti_range[0]
        
        # 光照参数
        scaled_points[:, 8] = points[:, 8] * (self.Sflux_range[1] - self.Sflux_range[0]) + self.Sflux_range[0]
        scaled_points[:, 9] = points[:, 9] * (self.alpha_sun_range[1] - self.alpha_sun_range[0]) + self.alpha_sun_range[0]
        scaled_points[:, 10] = points[:, 10] * (self.alpha_ram_range[1] - self.alpha_ram_range[0]) + self.alpha_ram_range[0]
        
        # 材料标识
        scaled_points[:, 11] = np.floor(points[:, 11] * self.n_materials).astype(float)
        scaled_points[:, 11] = np.clip(scaled_points[:, 11], 0, self.n_materials - 1)
        
        return scaled_points
    
    def generate_boundary_conditions(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成边界条件数据
        
        Args:
            n_points: 边界点数量
        
        Returns:
            boundary_points: 边界点 [n_points, input_dim]
            boundary_values: 边界值 [n_points, 1]
        """
        # 生成边界点（空间边界）
        boundary_points = []
        boundary_values = []
        
        # x边界
        for x_val in [self.x_range[0], self.x_range[1]]:
            n_boundary = n_points // 6
            points = self._uniform_sampling(n_boundary)
            points[:, 1] = x_val  # 固定x坐标
            boundary_points.append(points)
            # 边界条件：V = 0（接地）
            boundary_values.append(np.zeros((n_boundary, 1)))
        
        # y边界
        for y_val in [self.y_range[0], self.y_range[1]]:
            n_boundary = n_points // 6
            points = self._uniform_sampling(n_boundary)
            points[:, 2] = y_val  # 固定y坐标
            boundary_points.append(points)
            boundary_values.append(np.zeros((n_boundary, 1)))
        
        # z边界
        for z_val in [self.z_range[0], self.z_range[1]]:
            n_boundary = n_points // 6
            points = self._uniform_sampling(n_boundary)
            points[:, 3] = z_val  # 固定z坐标
            boundary_points.append(points)
            boundary_values.append(np.zeros((n_boundary, 1)))
        
        boundary_points = np.vstack(boundary_points)
        boundary_values = np.vstack(boundary_values)
        
        return (torch.tensor(boundary_points, dtype=torch.float32, device=self.device),
                torch.tensor(boundary_values, dtype=torch.float32, device=self.device))
    
    def generate_initial_conditions(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成初始条件数据
        
        Args:
            n_points: 初始点数量
        
        Returns:
            initial_points: 初始点 [n_points, input_dim]
            initial_values: 初始值 [n_points, 1]
        """
        # 生成初始时刻的点
        initial_points = self._uniform_sampling(n_points)
        initial_points[:, 0] = self.t_range[0]  # t = 0
        
        # 初始条件：V(x, t=0) = 0
        initial_values = np.zeros((n_points, 1))
        
        return (torch.tensor(initial_points, dtype=torch.float32, device=self.device),
                torch.tensor(initial_values, dtype=torch.float32, device=self.device))
    
    def generate_synthetic_data(self, n_points: int, noise_level: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成合成数据（模拟高保真仿真数据）
        
        Args:
            n_points: 数据点数量
            noise_level: 噪声水平
        
        Returns:
            data_points: 数据点 [n_points, input_dim]
            data_values: 数据值 [n_points, 1]
        """
        # 生成数据点
        data_points = self._uniform_sampling(n_points)
        
        # 使用简单的解析解作为"真值"（实际应用中这里是高保真仿真数据）
        data_values = self._analytical_solution(data_points)
        
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(data_values), data_values.shape)
            data_values += noise
        
        return (torch.tensor(data_points, dtype=torch.float32, device=self.device),
                torch.tensor(data_values, dtype=torch.float32, device=self.device))
    
    def _analytical_solution(self, points: np.ndarray) -> np.ndarray:
        """简单的解析解（用于生成合成数据）
        
        实际应用中，这里应该是高保真仿真数据
        """
        t, x, y, z = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        ne, Te = points[:, 4], points[:, 5]
        
        # 简单的时空依赖解
        V = (0.1 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * 
             np.exp(-t) * np.log10(ne / 1e7) * (Te / 3000))
        
        return V.reshape(-1, 1)
    
    def generate_physics_parameters(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从输入点生成物理参数字典
        
        Args:
            points: 输入点 [n_points, input_dim]
        
        Returns:
            params: 物理参数字典
        """
        batch_size = points.shape[0]
        
        # 从输入点提取参数
        ne = points[:, 4:5]  # 电子密度
        Te = points[:, 5:6]  # 电子温度
        ni = points[:, 6:7]  # 离子密度
        Ti = points[:, 7:8]  # 离子温度
        Sflux = points[:, 8:9]  # 太阳辐射通量
        alpha_sun = points[:, 9:10]  # 太阳入射角
        material_id = points[:, 11:12]  # 材料标识
        
        # 计算光电子参数
        Jph0 = 1e-6 * Sflux * torch.cos(alpha_sun)  # 饱和光电子电流密度
        Vph = 2.0 * torch.ones_like(Jph0)  # 等效光电子温度
        
        # 单位面积电容（简化模型）
        C = 1e-12 * torch.ones(batch_size, 1, device=self.device)  # F/m^2
        
        # 材料属性（简化）
        Mprop = torch.zeros(batch_size, 2, device=self.device)
        Mprop[:, 0] = material_id.squeeze()  # 材料ID
        Mprop[:, 1] = 1.0  # 材料属性参数
        
        params = {
            'ne': ne,
            'Te': Te,
            'ni': ni,
            'Ti': Ti,
            'Jph0': Jph0,
            'Vph': Vph,
            'C': C,
            'Mprop': Mprop
        }
        
        return params
    
    def visualize_data_distribution(self, points: torch.Tensor, save_path: Optional[str] = None):
        """可视化数据分布
        
        Args:
            points: 数据点
            save_path: 保存路径
        """
        points_np = points.cpu().numpy()
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        labels = ['t', 'x', 'y', 'z', 'ne', 'Te', 'ni', 'Ti', 'Sflux', 'alpha_sun', 'alpha_ram', 'material_id']
        
        for i in range(12):
            axes[i].hist(points_np[:, i], bins=50, alpha=0.7)
            axes[i].set_title(f'{labels[i]} distribution')
            axes[i].set_xlabel(labels[i])
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class DataNormalizer:
    """数据归一化器"""
    
    def __init__(self, method: str = 'minmax'):
        """
        Args:
            method: 归一化方法 ('minmax', 'zscore', 'robust')
        """
        self.method = method
        self.fitted = False
        self.stats = {}
    
    def fit(self, data: torch.Tensor):
        """拟合归一化参数
        
        Args:
            data: 训练数据 [n_samples, n_features]
        """
        if self.method == 'minmax':
            self.stats['min'] = torch.min(data, dim=0)[0]
            self.stats['max'] = torch.max(data, dim=0)[0]
            self.stats['range'] = self.stats['max'] - self.stats['min']
            # 避免除零
            self.stats['range'] = torch.where(self.stats['range'] == 0, 
                                            torch.ones_like(self.stats['range']), 
                                            self.stats['range'])
        elif self.method == 'zscore':
            self.stats['mean'] = torch.mean(data, dim=0)
            self.stats['std'] = torch.std(data, dim=0)
            # 避免除零
            self.stats['std'] = torch.where(self.stats['std'] == 0, 
                                          torch.ones_like(self.stats['std']), 
                                          self.stats['std'])
        elif self.method == 'robust':
            self.stats['median'] = torch.median(data, dim=0)[0]
            q75 = torch.quantile(data, 0.75, dim=0)
            q25 = torch.quantile(data, 0.25, dim=0)
            self.stats['iqr'] = q75 - q25
            # 避免除零
            self.stats['iqr'] = torch.where(self.stats['iqr'] == 0, 
                                          torch.ones_like(self.stats['iqr']), 
                                          self.stats['iqr'])
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """应用归一化
        
        Args:
            data: 输入数据
        
        Returns:
            normalized_data: 归一化后的数据
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if self.method == 'minmax':
            return (data - self.stats['min']) / self.stats['range']
        elif self.method == 'zscore':
            return (data - self.stats['mean']) / self.stats['std']
        elif self.method == 'robust':
            return (data - self.stats['median']) / self.stats['iqr']
    
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """逆归一化
        
        Args:
            normalized_data: 归一化的数据
        
        Returns:
            original_data: 原始尺度的数据
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        if self.method == 'minmax':
            return normalized_data * self.stats['range'] + self.stats['min']
        elif self.method == 'zscore':
            return normalized_data * self.stats['std'] + self.stats['mean']
        elif self.method == 'robust':
            return normalized_data * self.stats['iqr'] + self.stats['median']
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """拟合并应用归一化"""
        self.fit(data)
        return self.transform(data)