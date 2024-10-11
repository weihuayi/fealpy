from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike

class MBBBeam2dOneData:
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
    
    def force(self, points: TensorLike) -> TensorLike:

        val = bm.zeros(points.shape, dtype=points.dtype)
        # val[self.ny, 1] = -1
        val[0, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)
    
    def is_dirichlet_boundary_edge(self, edge_centers: TensorLike) -> TensorLike:

        left_edge = (edge_centers[:, 0] == 0.0)
        specific_edge = (edge_centers[:, 0] == self.nx) & (edge_centers[:, 1] == 0.5)
        
        result = left_edge | specific_edge

        return result
    
    def is_dirichlet_boundary_node(self) -> TensorLike:
        
        dirichlet_nodes = bm.zeros((self.nx+1)*(self.ny+1), dtype=bool)

        dirichlet_nodes[0:self.ny + 1] = True
        # dirichlet_nodes[(self.ny + 1) * self.nx] = True
        dirichlet_nodes[-1] = True

        return dirichlet_nodes
    
    def is_dirichlet_boundary_dof(self) -> TensorLike:
        
        direction_flags = bm.zeros(((self.nx + 1) * (self.ny + 1), 2), dtype=bool)

        direction_flags[0:self.ny+1, 0] = True
        # direction_flags[(self.ny + 1) * self.nx, 1] = True
        direction_flags[-1, 1] = True

        return direction_flags