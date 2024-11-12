from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from typing import Tuple, Callable
from builtins import list

class ShortCantilever2dData1:
    def __init__(self):
        """
        flip_direction = True
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.eps = 1e-12

    def domain(self, 
            xmin: float=0, xmax: float=4, 
            ymin: float=0, ymax: float=4) -> list:
        
        box = [xmin, xmax, ymin, ymax]

        return box
    
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
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
    
    def is_dirichlet_boundary_dof(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    def threshold(self) -> Callable:

        return self.is_dirichlet_boundary_dof
    
class ShortCantilever2dOneData:
    def __init__(self, nx: int, ny: int):
        """
        flip_direction = True
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.nx = nx
        self.ny = ny
    
    def domain(self):
        xmin, xmax = 0, 160
        ymin, ymax = 0, 100
        return [xmin, xmax, ymin, ymax]
    
    def force(self, points: TensorLike) -> TensorLike:

        val = bm.zeros(points.shape, dtype=points.dtype)
        val[-1, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)
    
    def is_dirichlet_boundary_edge(self, edge_centers: TensorLike) -> TensorLike:

        left_edge = (edge_centers[:, 0] == 0.0)

        return left_edge
    
    def is_dirichlet_node(self) -> TensorLike:
        
        dirichlet_nodes = bm.zeros((self.nx+1)*(self.ny+1), dtype=bool)

        dirichlet_nodes[0:self.ny + 1] = True

        return dirichlet_nodes
    
    def is_dirichlet_direction(self) -> TensorLike:
        
        direction_flags = bm.zeros(((self.nx + 1) * (self.ny + 1), 2), dtype=bool)

        direction_flags[0:self.ny, :] = True
        direction_flags[self.ny, 0] = True

        return direction_flags