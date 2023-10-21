import numpy as np
from typing import Optional, Tuple, Callable, Any, Union, List

from .mesh import UniformMesh2d

class MACNSSolver2d():
    """
    @brief 基于MAC 方法在笛卡尔交错网格上求解 NS 方程
    """

    def __init__(self, domain, nx=10, ny=10):
        """
        @brief 
        """
        self.dx = (domain[1] - domain[0])/nx
        self.dy = (domain[3] - domain[2])/ny
        self.umesh = UniformMesh2d(extent, h=(dx, dy), origin=origin)
        self.vmesh = UniformMesh2d()
        self.pmesh = UniformMesh2d()

    
