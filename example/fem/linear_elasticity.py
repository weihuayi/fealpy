
"""论文中带有线性位移真解的算例（应变和应力为常数）"""
from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian


class BoxDomainLinear3d():
    def __init__(self):
        self.eps = 1e-12

    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        val[..., 0] = 1e-3 * (2*x + y + z) / 2
        val[..., 1] = 1e-3 * (x + 2*y + z) / 2
        val[..., 2] = 1e-3 * (x + y + 2*z) / 2

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        result = self.solution(points)

        return result

node = bm.array([[0.249, 0.342, 0.192],
                [0.826, 0.288, 0.288],
                [0.850, 0.649, 0.263],
                [0.273, 0.750, 0.230],
                [0.320, 0.186, 0.643],
                [0.677, 0.305, 0.683],
                [0.788, 0.693, 0.644],
                [0.165, 0.745, 0.702],
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]],
            dtype=bm.float64)

cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                [0, 3, 2, 1, 8, 11, 10, 9],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [3, 7, 6, 2, 11, 15, 14, 10],
                [0, 1, 5, 4, 8, 9, 13, 12],
                [1, 2, 6, 5, 9, 10, 14, 13],
                [0, 4, 7, 3, 8, 12, 15, 11]],
                dtype=bm.int32)
mesh = HexahedronMesh(node, cell)


qf = mesh.quadrature_formula(2)
bcs, ws = qf.get_quadrature_points_and_weights()
vol = mesh.entity_measure('cell')

gphi = mesh.grad_shape_function(bcs, variables='x') # (NC, NQ, LDOF, GD)

gphi = gphi[..., [0, 4, 6, 2, 1, 5, 7, 3], :]

J = mesh.jacobi_matrix(bcs) # (NC, NQ, GD, GD)
detJ = bm.linalg.det(J)

# (NC, LDOF, GD)
BBar = bm.einsum('q, cqld, cq->cld', ws, gphi, detJ)/( 3 * vol[..., None, None]) 


NC = mesh.number_of_cells()
NQ = qf.number_of_quadrature_points()
GD = 3
LDOF = 8
B = bm.zeros((NC, NQ, 6, 3*LDOF), dtype=mesh.ftype)

B[..., 0, 0::3] =  2.0/3.0 * gphi[..., 0] + BBar[:, None, :, 0]
B[..., 0, 1::3] = -1.0/3.0 * gphi[..., 1] + BBar[:, None, :, 1]
B[..., 0, 2::3] = -1.0/3.0 * gphi[..., 2] + BBar[:, None, :, 2]

B[..., 1, 0::3] = -1.0/3.0 * gphi[..., 0] + BBar[:, None, :, 0]
B[..., 1, 1::3] =  2.0/3.0 * gphi[..., 1] + BBar[:, None, :, 1]
B[..., 1, 2::3] = -1.0/3.0 * gphi[..., 2] + BBar[:, None, :, 2]

B[..., 2, 0::3] = -1.0/3.0 * gphi[..., 0] + BBar[:, None, :, 0]
B[..., 2, 1::3] = -1.0/3.0 * gphi[..., 1] + BBar[:, None, :, 1]
B[..., 2, 2::3] =  2.0/3.0 * gphi[..., 2] + BBar[:, None, :, 2]


B[..., 3, 0::3] = gphi[..., 1]
B[..., 3, 1::3] = gphi[..., 0]

B[..., 4, 0::3] = gphi[..., 2]
B[..., 4, 2::3] = gphi[..., 0]

B[..., 5, 1::3] = gphi[..., 2]
B[..., 5, 2::3] = gphi[..., 1]



# 刚度矩阵
E = 2.1e5 
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                    [lam, 2 * mu + lam, lam, 0, 0, 0],
                    [lam, lam, 2 * mu + lam, 0, 0, 0],
                    [0, 0, 0, mu, 0, 0],
                    [0, 0, 0, 0, mu, 0],
                    [0, 0, 0, 0, 0, mu]], dtype=mesh.ftype)

K = bm.einsum('q, cqik, ij, cqjl, cq->ckl', ws, B, D, B, detJ)
print("bcs:\n", bcs)
print("ws:\n", ws)
print("vol:\n", vol)
print("vol.sum():\n", vol.sum())
print('gphi.shape:', gphi.shape)
print("B.shape:\n", B.shape)
print("J.shape:\n", J.shape)
print("detJ.shape:\n", detJ.shape)
print("lam:\n", lam)
print("mu:\n", mu)
print("D:\n", D)
print("K.shape:\n", K.shape) 
print("K:\n", K[0])
