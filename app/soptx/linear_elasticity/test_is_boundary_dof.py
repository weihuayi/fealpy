from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.decorator import cartesian


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
        self.eps = 1e-12

    def domain(self):
        return [0, 1, 0, 1]
    
    def force(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]

        index1 = (
            (bm.abs(x - self.domain[0]) < self.eps) & 
            (bm.abs(y - self.domain[1]) < self.eps)
        )
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[index1, 1] = -1

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
    
    def is_dirichlet_boundary_dof(self, points):
        x = points[..., 0]
        y = points[..., 1]

        cond1 = (bm.abs(x - 0) < self.eps) & (bm.abs(y - 0) < self.eps)
        cond2 = (bm.abs(x - 0) < self.eps) & (bm.abs(y - 1) < self.eps)
        cond3 = (bm.abs(x - 1) < self.eps) & (bm.abs(y - 0) < self.eps)
        cond4 = (bm.abs(x - 1) < self.eps) & (bm.abs(y - 1) < self.eps)

        result = cond1 | cond2 | cond3 | cond4

        return result
    

pde = MBBBeam2dOneData(nx=4, ny=4)

bm.set_backend('numpy')


mesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=4, ny=4)

space = LagrangeFESpace(mesh, p=1, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
bm.jacfwd()

isBdDof = tensor_space.is_boundary_dof(threshold=(pde.is_dirichlet_boundary_dof, ))
print("-----------------")
