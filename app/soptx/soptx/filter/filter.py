from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor

from .matrix import FilterMatrix

class FilterType(IntEnum):
    """滤波器类型枚举"""
    SENSITIVITY = 0  # 灵敏度滤波
    DENSITY = 1      # 密度滤波
    HEAVISIDE = 2    # Heaviside投影滤波

@dataclass
class FilterConfig:
    """滤波器配置类"""
    filter_type: FilterType  # 使用枚举类型
    filter_radius: float

    def __post_init__(self):
        if self.filter_radius <= 0:
            raise ValueError("Filter radius must be positive")
        if not isinstance(self.filter_type, (FilterType, int)):
            raise ValueError("Filter type must be a FilterType enum or valid integer")
        if isinstance(self.filter_type, int):
            self.filter_type = FilterType(self.filter_type)

class Filter:
    """滤波器类，负责滤波矩阵计算和灵敏度修改"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._H: Optional[COOTensor] = None  
        self._Hs: Optional[TensorLike] = None
        self._mesh = None

        self._rho_tilde = None  # 存储中间密度场(对 Heaviside 投影有用)

    def initialize(self, mesh) -> None:
        """根据网格初始化滤波矩阵"""
        self._mesh = mesh
        self._H, self._Hs = FilterMatrix.create_filter_matrix(mesh, self.config.filter_radius)

    @property
    def get_intermediate_density(self) -> Optional[TensorLike]:
        """获取中间密度场"""
        return self._rho_tilde

    @property
    def get_H(self) -> COOTensor:
        """滤波矩阵"""
        if self._H is None:
            raise ValueError("Filter matrix is not initialized")
        return self._H

    @property
    def get_Hs(self) -> TensorLike:
        """滤波矩阵行和向量"""
        if self._Hs is None:
            raise ValueError("Filter scaling vector is not initialized")
        return self._Hs
    
    def get_physical_density(self, 
                        density: TensorLike,
                        filter_params: Optional[dict] = None) -> TensorLike:
        """
        获取物理密度场. 
        只有在 Heaviside 投影滤波时才会对密度进行变换, 其他情况直接返回输入密度.

        Parameters
        ----------
        density : TensorLike
            原始密度场
        filter_params : Optional[dict]
            过滤器参数 (Heaviside 投影需要)
            - beta: 投影参数

        Returns
        -------
        TensorLike
            物理密度场
        """
        if self.config.filter_type == FilterType.HEAVISIDE:
            if filter_params is None or 'beta' not in filter_params:
                raise ValueError("Heaviside projection requires 'beta' parameter")

            beta = filter_params['beta']
            # 计算并存储中间密度场
            self._rho_tilde = density
            # 应用 Heaviside 投影
            physical_density = (1 - bm.exp(-beta * self._rho_tilde) + 
                            self._rho_tilde * bm.exp(-beta))

            return physical_density
        else:
            # 对于灵敏度滤波和密度滤波，初始物理密度等于设计密度
            return density

        
    def filter_sensitivity(self,
                         gradient: TensorLike,
                         design_vars: TensorLike,
                         gradient_type: Literal['objective', 'constraint'],
                         filter_params: Optional[dict] = None) -> TensorLike:
        """
        应用滤波器修改灵敏度
        
        Parameters
        - gradient : 原始梯度
        - design_vars : 设计变量
        - gradient_type : 梯度类型, 用于区分目标函数梯度和约束函数梯度
        - filter_params : 滤波器参数
            
        Returns
        """
        cell_measure = self._mesh.entity_measure('cell')

        if self.config.filter_type == FilterType.SENSITIVITY:
            # 灵敏度滤波只修改目标函数的梯度
            if gradient_type == 'objective':
                # 计算密度加权的灵敏度
                weighted_gradient = bm.einsum('c, c -> c', design_vars, gradient)
                # 应用滤波矩阵
                filtered_gradient = self._H.matmul(weighted_gradient)
                 # 计算修正因子
                correction_factor = self._Hs * bm.maximum(bm.tensor(0.001, dtype=bm.float64), design_vars)
                # 返回最终修改的灵敏度
                modified_gradient = filtered_gradient / correction_factor

                return modified_gradient
            else:
                # 约束函数的梯度不做修改
                return gradient
                
        elif self.config.filter_type == FilterType.DENSITY:
            # 密度滤波对两种梯度都进行修改
            # 计算单元度量加权的灵敏度
            weighted_gradient = gradient * cell_measure
            # 计算标准化因子    
            normalization_factor = self._H.matmul(cell_measure)
            # 应用滤波矩阵
            filtered_gradient = self._H.matmul(weighted_gradient / normalization_factor)

            return filtered_gradient
            
        elif self.config.filter_type == FilterType.HEAVISIDE:
            if self._rho_tilde is None:
                raise ValueError("Intermediate density not available. Call get_physical_density first to initialize intermediate density field.")

            if not filter_params or 'beta' not in filter_params:
                raise ValueError("Heaviside projection requires beta parameter")
            
            beta = filter_params['beta']
            # 计算投影导数
            dx = beta * bm.exp(-beta * self._rho_tilde) + bm.exp(-beta)
            # 修改梯度
            weighted_gradient = gradient * dx * cell_measure
            filtered_gradient = self._H.matmul(weighted_gradient)
            normalization_factor = self._H.matmul(cell_measure)
            return filtered_gradient / normalization_factor
    
    def filter_density(self,
                density: TensorLike,
                filter_params: Optional[dict] = None) -> TensorLike:
        """
        对密度进行滤波

        Parameters
        - density : 原始密度场
        - filter_params : 过滤器参数, 对于 Heaviside 滤波需要:
            - beta: 投影参数

        Returns
        - 过滤后的物理密度场
        """
        # 获取单元度量
        cell_measure = self._mesh.entity_measure('cell')

        if self.config.filter_type == FilterType.SENSITIVITY:
            # 灵敏度滤波时，物理密度等于设计密度
            return density  

        elif self.config.filter_type == FilterType.DENSITY:
            # 计算加权密度
            weighted_density = density * cell_measure
            # 应用滤波矩阵
            filtered_density = self._H.matmul(weighted_density)
            # 计算标准化因子
            normalization_factor = self._H.matmul(cell_measure)
            # 返回标准化后的密度
            physical_density = filtered_density / normalization_factor

            return physical_density

        elif self.config.filter_type == FilterType.HEAVISIDE:
            if filter_params is None or 'beta' not in filter_params:
                raise ValueError("Heaviside projection requires beta parameter")

            beta = filter_params['beta']

            # 计算并存储中间密度场
            weighted_density = density * cell_measure
            filtered_density = self._H.matmul(weighted_density)
            normalization = self._H.matmul(cell_measure)
            self._rho_tilde = filtered_density / normalization

            # 应用 Heaviside 投影
            physical_density = (1 - bm.exp(-beta * self._rho_tilde) + 
                              self._rho_tilde * bm.exp(-beta))
            
            return physical_density

