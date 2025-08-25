import numpy as np
import torch
import deepxde as dde
from typing import Dict, List, Tuple, Callable, Optional

class BoundaryConditionManager:
    """边界条件和初始条件管理器"""
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]]):
        self.domain_bounds = domain_bounds
        self.boundary_conditions = []
        self.initial_conditions = []
        self.data_constraints = []
    
    def add_dirichlet_bc(self, boundary_func: Callable, 
                        value_func: Callable, 
                        description: str = ""):
        """添加Dirichlet边界条件
        
        Args:
            boundary_func: 边界判断函数
            value_func: 边界值函数
            description: 边界条件描述
        """
        bc_config = {
            'type': 'dirichlet',
            'boundary': boundary_func,
            'value': value_func,
            'description': description
        }
        self.boundary_conditions.append(bc_config)
    
    def add_neumann_bc(self, boundary_func: Callable, 
                      value_func: Callable, 
                      description: str = ""):
        """添加Neumann边界条件
        
        Args:
            boundary_func: 边界判断函数
            value_func: 边界值函数 (法向导数)
            description: 边界条件描述
        """
        bc_config = {
            'type': 'neumann',
            'boundary': boundary_func,
            'value': value_func,
            'description': description
        }
        self.boundary_conditions.append(bc_config)
    
    def add_initial_condition(self, initial_func: Callable, 
                            description: str = ""):
        """添加初始条件
        
        Args:
            initial_func: 初始条件函数
            description: 初始条件描述
        """
        ic_config = {
            'type': 'initial',
            'boundary': lambda x, on_initial: np.isclose(x[:, 0], self.domain_bounds['t'][0]),
            'value': initial_func,
            'description': description
        }
        self.initial_conditions.append(ic_config)
    
    def add_data_constraint(self, X_data: np.ndarray, y_data: np.ndarray, 
                          description: str = ""):
        """添加数据约束
        
        Args:
            X_data: 输入数据点
            y_data: 对应的输出数据
            description: 数据约束描述
        """
        data_config = {
            'X_data': X_data,
            'y_data': y_data,
            'description': description
        }
        self.data_constraints.append(data_config)
    
    def get_all_conditions(self) -> List[Dict]:
        """获取所有边界条件和初始条件"""
        return self.boundary_conditions + self.initial_conditions
    
    def get_data_constraints(self) -> List[Dict]:
        """获取所有数据约束"""
        return self.data_constraints

class StandardBoundaryConditions:
    """标准边界条件定义"""
    
    @staticmethod
    def grounded_surface(domain_bounds: Dict[str, Tuple[float, float]]) -> Callable:
        """接地表面边界条件 (V = 0)
        
        Args:
            domain_bounds: 域边界
        
        Returns:
            边界条件函数
        """
        def boundary_func(x, on_boundary):
            # 在空间边界上的接地条件
            x_min, x_max = domain_bounds['x']
            y_min, y_max = domain_bounds.get('y', (0, 1))
            z_min, z_max = domain_bounds.get('z', (0, 1))
            
            return on_boundary and (
                np.isclose(x[:, 1], x_min) or np.isclose(x[:, 1], x_max) or
                np.isclose(x[:, 2], y_min) or np.isclose(x[:, 2], y_max) or
                np.isclose(x[:, 3], z_min) or np.isclose(x[:, 3], z_max)
            )
        
        def value_func(x):
            return np.zeros((len(x), 1))
        
        return boundary_func, value_func
    
    @staticmethod
    def floating_potential(V_float: float) -> Tuple[Callable, Callable]:
        """浮动电位边界条件
        
        Args:
            V_float: 浮动电位值
        
        Returns:
            边界条件函数
        """
        def boundary_func(x, on_boundary):
            # 可以根据具体需求定义哪些边界使用浮动电位
            return on_boundary
        
        def value_func(x):
            return np.full((len(x), 1), V_float)
        
        return boundary_func, value_func
    
    @staticmethod
    def periodic_boundary(domain_bounds: Dict[str, Tuple[float, float]], 
                         axis: str = 'x') -> List[Dict]:
        """周期性边界条件
        
        Args:
            domain_bounds: 域边界
            axis: 周期性轴 ('x', 'y', 'z')
        
        Returns:
            周期性边界条件配置列表
        """
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        axis_idx = axis_map[axis]
        axis_min, axis_max = domain_bounds[axis]
        
        def left_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[:, axis_idx], axis_min)
        
        def right_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[:, axis_idx], axis_max)
        
        def periodic_value(x):
            return np.zeros((len(x), 1))  # 周期性条件: V(left) - V(right) = 0
        
        return [
            {
                'type': 'dirichlet',
                'boundary': left_boundary,
                'value': periodic_value,
                'description': f'Periodic BC - {axis} left'
            },
            {
                'type': 'dirichlet', 
                'boundary': right_boundary,
                'value': periodic_value,
                'description': f'Periodic BC - {axis} right'
            }
        ]
    
    @staticmethod
    def zero_flux_boundary(domain_bounds: Dict[str, Tuple[float, float]], 
                          axis: str = 'x') -> Tuple[Callable, Callable]:
        """零通量边界条件 (∂V/∂n = 0)
        
        Args:
            domain_bounds: 域边界
            axis: 法向轴
        
        Returns:
            Neumann边界条件函数
        """
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        axis_idx = axis_map[axis]
        axis_min, axis_max = domain_bounds[axis]
        
        def boundary_func(x, on_boundary):
            return on_boundary and (
                np.isclose(x[:, axis_idx], axis_min) or 
                np.isclose(x[:, axis_idx], axis_max)
            )
        
        def value_func(x):
            return np.zeros((len(x), 1))
        
        return boundary_func, value_func

class InitialConditions:
    """初始条件定义"""
    
    @staticmethod
    def zero_initial_potential() -> Callable:
        """零初始电位
        
        Returns:
            初始条件函数
        """
        def initial_func(x):
            return np.zeros((len(x), 1))
        
        return initial_func
    
    @staticmethod
    def constant_initial_potential(V0: float) -> Callable:
        """常数初始电位
        
        Args:
            V0: 初始电位值
        
        Returns:
            初始条件函数
        """
        def initial_func(x):
            return np.full((len(x), 1), V0)
        
        return initial_func
    
    @staticmethod
    def gaussian_initial_potential(center: List[float], 
                                  amplitude: float, 
                                  width: float) -> Callable:
        """高斯分布初始电位
        
        Args:
            center: 高斯中心位置 [x, y, z]
            amplitude: 振幅
            width: 宽度参数
        
        Returns:
            初始条件函数
        """
        def initial_func(x):
            # x的格式: [t, x, y, z, ...]
            spatial_coords = x[:, 1:4]  # 提取空间坐标
            center_array = np.array(center)
            
            # 计算到中心的距离
            distances = np.linalg.norm(spatial_coords - center_array, axis=1)
            
            # 高斯分布
            values = amplitude * np.exp(-distances**2 / (2 * width**2))
            
            return values.reshape(-1, 1)
        
        return initial_func
    
    @staticmethod
    def sinusoidal_initial_potential(amplitude: float, 
                                   frequency: List[float],
                                   phase: float = 0.0) -> Callable:
        """正弦分布初始电位
        
        Args:
            amplitude: 振幅
            frequency: 各方向频率 [fx, fy, fz]
            phase: 相位
        
        Returns:
            初始条件函数
        """
        def initial_func(x):
            # x的格式: [t, x, y, z, ...]
            spatial_coords = x[:, 1:4]  # 提取空间坐标
            
            # 计算正弦值
            values = amplitude * np.sin(
                2 * np.pi * (
                    frequency[0] * spatial_coords[:, 0] +
                    frequency[1] * spatial_coords[:, 1] +
                    frequency[2] * spatial_coords[:, 2]
                ) + phase
            )
            
            return values.reshape(-1, 1)
        
        return initial_func

class DataConstraintGenerator:
    """数据约束生成器"""
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]]):
        self.domain_bounds = domain_bounds
    
    def generate_synthetic_data(self, 
                              num_points: int,
                              noise_level: float = 0.0,
                              data_type: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """生成合成数据
        
        Args:
            num_points: 数据点数量
            noise_level: 噪声水平
            data_type: 数据类型 ('random', 'grid', 'boundary')
        
        Returns:
            X_data, y_data: 输入数据和对应的输出数据
        """
        if data_type == 'random':
            X_data = self._generate_random_points(num_points)
        elif data_type == 'grid':
            X_data = self._generate_grid_points(num_points)
        elif data_type == 'boundary':
            X_data = self._generate_boundary_points(num_points)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # 生成对应的输出数据 (这里使用简单的解析解作为示例)
        y_data = self._analytical_solution(X_data)
        
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, y_data.shape)
            y_data += noise
        
        return X_data, y_data
    
    def _generate_random_points(self, num_points: int) -> np.ndarray:
        """生成随机采样点"""
        points = []
        
        for key, (min_val, max_val) in self.domain_bounds.items():
            if key in ['t', 'x', 'y', 'z', 'n_e', 'T_e', 'n_i', 'T_i', 
                      'S_flux', 'alpha_sun', 'alpha_ram', 'material_id']:
                if key in ['n_e', 'n_i']:  # 对数尺度采样
                    samples = np.random.uniform(
                        np.log10(min_val), np.log10(max_val), num_points
                    )
                    samples = 10**samples
                else:
                    samples = np.random.uniform(min_val, max_val, num_points)
                points.append(samples)
        
        return np.column_stack(points)
    
    def _generate_grid_points(self, num_points: int) -> np.ndarray:
        """生成网格采样点"""
        # 简化版本：只考虑时空坐标
        t_bounds = self.domain_bounds['t']
        x_bounds = self.domain_bounds['x']
        
        # 计算每个维度的点数
        points_per_dim = int(np.sqrt(num_points))
        
        t_vals = np.linspace(t_bounds[0], t_bounds[1], points_per_dim)
        x_vals = np.linspace(x_bounds[0], x_bounds[1], points_per_dim)
        
        T, X = np.meshgrid(t_vals, x_vals)
        
        # 为其他维度设置默认值
        points = np.column_stack([
            T.flatten(),
            X.flatten(),
            np.zeros(T.size),  # y
            np.zeros(T.size),  # z
            np.full(T.size, 1e9),  # n_e
            np.full(T.size, 10000),  # T_e
            np.full(T.size, 1e9),  # n_i
            np.full(T.size, 1000),  # T_i
            np.full(T.size, 1000),  # S_flux
            np.full(T.size, 0),  # alpha_sun
            np.full(T.size, 0),  # alpha_ram
            np.zeros(T.size)  # material_id
        ])
        
        return points[:num_points]
    
    def _generate_boundary_points(self, num_points: int) -> np.ndarray:
        """生成边界采样点"""
        # 在边界上生成点
        points = []
        
        # 时间边界
        t_min, t_max = self.domain_bounds['t']
        x_min, x_max = self.domain_bounds['x']
        
        # 初始时刻的点
        for i in range(num_points // 2):
            point = [t_min]  # t = t_min
            point.append(np.random.uniform(x_min, x_max))  # 随机x
            
            # 其他维度的随机值
            for key, (min_val, max_val) in self.domain_bounds.items():
                if key not in ['t', 'x']:
                    if key in ['n_e', 'n_i']:
                        val = 10**np.random.uniform(np.log10(min_val), np.log10(max_val))
                    else:
                        val = np.random.uniform(min_val, max_val)
                    point.append(val)
            
            points.append(point)
        
        # 空间边界的点
        for i in range(num_points - num_points // 2):
            point = [np.random.uniform(t_min, t_max)]  # 随机t
            point.append(x_min if i % 2 == 0 else x_max)  # 边界x
            
            # 其他维度的随机值
            for key, (min_val, max_val) in self.domain_bounds.items():
                if key not in ['t', 'x']:
                    if key in ['n_e', 'n_i']:
                        val = 10**np.random.uniform(np.log10(min_val), np.log10(max_val))
                    else:
                        val = np.random.uniform(min_val, max_val)
                    point.append(val)
            
            points.append(point)
        
        return np.array(points)
    
    def _analytical_solution(self, X: np.ndarray) -> np.ndarray:
        """简单的解析解 (用于测试)
        
        Args:
            X: 输入点
        
        Returns:
            对应的解析解值
        """
        # 简单的时空依赖解析解
        t = X[:, 0]
        x = X[:, 1]
        
        # 示例：衰减振荡解
        solution = np.exp(-0.1 * t) * np.sin(np.pi * x) * np.cos(2 * np.pi * t)
        
        return solution.reshape(-1, 1)
    
    def load_experimental_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载实验数据
        
        Args:
            file_path: 数据文件路径
        
        Returns:
            X_data, y_data: 输入数据和输出数据
        """
        # 这里可以根据具体的数据格式进行实现
        # 例如从CSV、HDF5、MAT文件等加载数据
        try:
            if file_path.endswith('.csv'):
                import pandas as pd
                data = pd.read_csv(file_path)
                # 假设最后一列是输出，其余是输入
                X_data = data.iloc[:, :-1].values
                y_data = data.iloc[:, -1:].values
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                X_data = data['X']
                y_data = data['y']
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            return X_data, y_data
        
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None, None