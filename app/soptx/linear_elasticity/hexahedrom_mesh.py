from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.decorator import cartesian
from fealpy.experimental.mesh import HexahedronMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from fealpy.experimental.fem.bilinear_form import BilinearForm

from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg


import argparse

class BoxDomain1Data3d():
    def __init__(self):
        pass
        
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=points.dtype)
        
        val[..., 0] = 2 * x**3 - 3 * x * y**2 - 3 * x * z**2
        val[..., 1] = 2 * y**3 - 3 * y * x**2 - 3 * y * z**2
        val[..., 2] = 2 * z**3 - 3 * z * y**2 - 3 * z * x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):

        val = bm.zeros(points.shape, dtype=points.dtype)
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    
class BoxDomain2Data3d():
    def __init__(self):
        pass
        
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=points.dtype)
        S = bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * bm.sin(bm.pi * z)
        
        val[..., 0] = 10 * S
        val[..., 1] = 10 * S
        val[..., 2] = 10 * S

        return val

    @cartesian
    def source(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=points.dtype)
        S = bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * bm.sin(bm.pi * z)
        lam = 1
        mu = 1
        pi = bm.pi
        val[..., 0] = -10*pi**2 * ((lam+mu) * bm.cos(pi*x) * bm.sin(pi*y+pi*z) - (lam+4*mu)*S)
        val[..., 1] = -10*pi**2 * ((lam+mu) * bm.sin(pi*y) * bm.cos(pi*x+pi*z) - (lam+4*mu)*S)
        val[..., 2] = -10*pi**2 * ((lam+mu) * bm.sin(pi*z) * bm.cos(pi*x+pi*y) - (lam+4*mu)*S)
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    

parser = argparse.ArgumentParser(description="HexahedronMesh 上的任意次 Lagrange 有限元空间的线性弹性问题求解.")
parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')
parser.add_argument('--degree', 
                    default=3, type=int, 
                    help='Lagrange 有限元空间的次数, 默认为 1 次.')
parser.add_argument('--nx', 
                    default=2, type=int, 
                    help='x 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--ny',
                    default=2, type=int,
                    help='y 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--nz',
                    default=2, type=int,
                    help='z 方向的初始网格单元数, 默认为 2.')
args = parser.parse_args()

pde = BoxDomain2Data3d()
args = parser.parse_args()

bm.set_backend(args.backend)
nx, ny, nz = args.nx, args.ny, args.nz
mesh = HexahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz)

p = args.degree

maxit = 3
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 3))
    tgdof = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', lame_lambda=1, shear_modulus=1)
    integrator = LinearElasticIntegrator(material=linear_elastic_material, q=tensor_space.p+3)
    KE = integrator.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly()

    F = tensor_space.interpolate(pde.source)

    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64)

    uh_bd, isDDof = tensor_space.boundary_interpolate(gD=pde.dirichlet, uh=uh_bd, threshold=None)

    F = F - K.matmul(uh_bd)
    F[isDDof] = uh_bd[isDDof]

    indices = K.indices()
    new_values = bm.copy(K.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0

    K = COOTensor(indices, new_values, K.sparse_shape)
    index, = bm.nonzero(isDDof)
    one_values = bm.ones(len(index), **K.values_context())
    one_indices = bm.stack([index, index], axis=0)
    K1 = COOTensor(one_indices, one_values, K.sparse_shape)
    K = K.add(K1).coalesce()

    uh = tensor_space.function()

    uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)

    u_exact = tensor_space.interpolate(pde.solution)
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh - u_exact)**2 * (1 / tgdof)))
    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)

    print("errorMatrix:", errorMatrix)

    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("order:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))