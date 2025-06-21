# TODO 删除
from dataclasses import dataclass

from typing import Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.material.elastic_material import LinearElasticMaterial

from .base import MaterialInterpolation

@dataclass
class ElasticMaterialConfig:
    """Configuration class for elastic material properties."""
    
    elastic_modulus: float = 1.0
    minimal_modulus: float = 1e-9
    poisson_ratio: float = 0.3
    plane_assumption: Literal["plane_stress", "plane_strain", "3d"] = "plane_stress"

class ElasticMaterialInstance(LinearElasticMaterial):
    """具有特定杨氏模量的弹性材料实例"""
    def __init__(self, E: TensorLike, config: ElasticMaterialConfig):
        super().__init__(
            name="ElasticMaterial",
            elastic_modulus=1.0,                # 基础值，实际值由 _E 控制
            poisson_ratio=config.poisson_ratio,
            hypo=config.plane_assumption
        )
        self._E = E
        self.config = config

    @property
    def elastic_modulus(self) -> TensorLike:
        """获取当前的杨氏模量场"""
        return self._E
        
    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """计算弹性矩阵"""
        base_D = super().elastic_matrix()

        # 处理不同类型的张量
        if len(self._E.shape) > 0:
            D = bm.einsum('b, ijkl -> bjkl', self._E, base_D)
        else:
            D = self._E * base_D
   
        return D
    
class ElasticMaterialProperties:
    """材料属性计算类，负责材料的插值计算"""
    def __init__(self, 
                config: ElasticMaterialConfig, 
                interpolation_model: MaterialInterpolation):
        
        if not isinstance(config, ElasticMaterialConfig):
            raise TypeError("'config' must be an instance of ElasticMaterialConfig")
        
        if interpolation_model is None or not isinstance(interpolation_model, MaterialInterpolation):
            raise TypeError("'interpolation_model' must be an instance of MaterialInterpolation")
            
        self.config = config
        self.interpolation_model = interpolation_model

    def calculate_elastic_modulus(self, density: TensorLike) -> TensorLike:
        """根据密度计算杨氏模量"""
        E = self.interpolation_model.calculate_property(
                density,
                self.config.elastic_modulus,
                self.config.minimal_modulus,
                self.interpolation_model.penalty_factor
            )
        return E

    def calculate_elastic_modulus_derivative(self, density: TensorLike) -> TensorLike:
        """计算杨氏模量对密度的导数"""
        dE = self.interpolation_model.calculate_property_derivative(
                density,
                self.config.elastic_modulus,
                self.config.minimal_modulus,
                self.interpolation_model.penalty_factor
            )
        return dE

    def get_base_material(self) -> ElasticMaterialInstance:
        """获取基础材料实例 (E=1)"""
        E = bm.ones(1, dtype=bm.float64)
        return ElasticMaterialInstance(E, self.config)