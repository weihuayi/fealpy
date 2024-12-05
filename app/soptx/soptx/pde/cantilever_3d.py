from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from typing import Tuple, Callable


class Cantilever3dData1:
    def __init__(self,
                xmin: float=0, xmax: float=60, 
                ymin: float=0, ymax: float=20,
                zmin: float=0, zmax: float=4):
        """
        flip_direction = 'y'
           1------- 5
         / |       /|
        3 ------- 7 |
        |  |      | |
        |  0------|-4
        | /       |/
        2 ------- 6
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, 
               self.ymin, self.ymax, 
               self.zmin, self.zmax]

        return box
    
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)
        )
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[coord, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)
    
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)

    