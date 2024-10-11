from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike

class Cantilever3dOneData:
    def __init__(self, nx: int, ny: int, nz: int):
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
        self.nx = nx
        self.ny = ny
        self.nz = nz
    
    def force(self, points: TensorLike) -> TensorLike:

        val = bm.zeros(points.shape, dtype=points.dtype)

        val[-(self.nz+1):, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)
    
    def is_dirichlet_boundary_face(self, face_centers: TensorLike) -> TensorLike:

        left_face = (face_centers[:, 0] == 0.0)

        return left_face
    