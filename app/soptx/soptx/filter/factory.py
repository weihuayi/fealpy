from typing import Optional, Union
from fealpy.mesh.mesh_base import Mesh

from .types import FilterType, FilterProperties
from .matrix import FilterMatrixBuilder

def create_filter_properties(
    mesh: Mesh,
    filter_type: Optional[Union[int, str]] = None,
    filter_rmin: Optional[float] = None
) -> Optional[FilterProperties]:
    """创建过滤器属性
    
    Args:
        mesh: 网格对象
        filter_type: 过滤器类型(整数、字符串或None)
        filter_rmin: 过滤半径
        
    Returns:
        FilterProperties or None: 过滤器属性对象，如果不需要过滤则返回None
        
    Raises:
        ValueError: 当参数组合无效时
        TypeError: 当参数类型错误时
    """
    if filter_type is None:
        if filter_rmin is not None:
            raise ValueError("filter_rmin must be None when filter_type is None")
        return None
        
    if filter_rmin is None:
        raise ValueError("filter_rmin required when filter_type is specified")
    
    # 解析过滤器类型
    if isinstance(filter_type, str):
        ft = FilterType.from_string(filter_type)
    elif isinstance(filter_type, int):
        try:
            ft = FilterType(filter_type)
        except ValueError:
            raise ValueError(f"Invalid filter type integer: {filter_type}")
    else:
        raise TypeError("filter_type must be int, str, or None")
    
    # 构建过滤矩阵
    H, Hs = FilterMatrixBuilder.build(mesh, filter_rmin)
    
    return FilterProperties(
                            mesh=mesh,
                            rmin=filter_rmin,
                            filter_type=ft,
                            H=H,
                            Hs=Hs
                        )