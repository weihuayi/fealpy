from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike

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