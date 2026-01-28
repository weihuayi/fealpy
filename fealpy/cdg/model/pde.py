from ...backend import TensorLike
from ...decorator import cartesian

class HarmonicMapPDE:
    """
    调和映射 PDE 数据类
    对应方程: -Δu = f
    """
    def __init__(self):
        pass

    @cartesian
    def source(self, p: TensorLike) -> float:
        """源项 f，对于调和映射通常为 0"""
        return 0.0