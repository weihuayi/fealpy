from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from typing import Tuple, Callable
from builtins import list

class MBBBeam2dData1:
    '''
    模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    def __init__(self, 
                xmin: float=0, xmax: float=60, 
                ymin: float=0, ymax: float=20):
        """
        flip_direction = True
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[0]) < self.eps) & 
            (bm.abs(y - domain[3]) < self.eps)
        )
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[coord, 1] = -1

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = ((bm.abs(x - domain[1]) < self.eps) &
                 (bm.abs(y - domain[0]) < self.eps))
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
