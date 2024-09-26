from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike

class Cantilever3dOneData:
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
    
    def domain(self):
        xmin, xmax = 0, 4
        ymin, ymax = 0, 1
        zmin, zmax = 0, 2
        return [xmin, xmax, ymin, ymax, zmin, zmax]
    
    def force(self, points: TensorLike) -> TensorLike:

        val = bm.zeros(points.shape, dtype=points.dtype)
        val[-(self.nz+1):, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)
    
    def is_dirichlet_boundary_face(self, face_centers: TensorLike) -> TensorLike:

        left_face = (face_centers[:, 0] == 0.0)

        return left_face