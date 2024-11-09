from enum import Enum
from dataclasses import dataclass
from typing import Optional
from fealpy.typing import TensorLike
from fealpy.mesh.mesh_base import Mesh

class FilterType(Enum):
    """过滤器类型枚举"""
    NONE = -1
    SENSITIVITY = 0
    DENSITY = 1
    HEAVISIDE = 2
    
    @classmethod
    def from_string(cls, filter_type: str) -> 'FilterType':
        """从字符串创建过滤器类型"""
        mapping = {
            'sens': cls.SENSITIVITY,
            'dens': cls.DENSITY,
            'heaviside': cls.HEAVISIDE
        }
        ft = mapping.get(filter_type.lower())
        if ft is None:
            raise ValueError(
                f"Invalid filter type string: '{filter_type}'. "
                f"Valid options are: {list(mapping.keys())}"
            )
        return ft
    
@dataclass
class FilterProperties:
    """过滤器基本属性"""
    mesh: Mesh
    rmin: float
    filter_type: FilterType
    H: Optional[TensorLike] = None    # 过滤矩阵
    Hs: Optional[TensorLike] = None   # 缩放向量