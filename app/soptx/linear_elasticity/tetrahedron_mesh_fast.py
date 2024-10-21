from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import TetrahedronMesh

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm

from fealpy.decorator import cartesian

from fealpy.sparse import COOTensor

from fealpy.solver import cg

from app.soptx.soptx.utils.timer import timer

import argparse


class BoxDomainPolyUnloaded3d():
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
        val[..., 0] = 2*x**3 - 3*x*y**2 - 3*x*z**2
        val[..., 1] = 2*y**3 - 3*y*x**2 - 3*y*z**2
        val[..., 2] = 2*z**3 - 3*z*y**2 - 3*z*x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, dtype=points.dtype)
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    
class BoxDomainPolyLoaded3d():
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def source(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=bm.float64)
        mu = 1
        factor1 = -400 * mu * (2 * y - 1) * (2 * z - 1)
        term1 = 3 * (x ** 2 - x) ** 2 * (y ** 2 - y + z ** 2 - z)
        term2 = (1 - 6 * x + 6 * x ** 2) * (y ** 2 - y) * (z ** 2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = 200 * mu * (2 * x - 1) * (2 * z - 1)
        term1 = 3 * (y ** 2 - y) ** 2 * (x ** 2 - x + z ** 2 - z)
        term2 = (1 - 6 * y + 6 * y ** 2) * (x ** 2 - x) * (z ** 2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = 200 * mu * (2 * x - 1) * (2 * y - 1)
        term1 = 3 * (z ** 2 - z) ** 2 * (x ** 2 - x + y ** 2 - y)
        term2 = (1 - 6 * z + 6 * z ** 2) * (x ** 2 - x) * (y ** 2 - y)
        val[..., 2] = factor3 * (term1 + term2)

        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=bm.float64)

        mu = 1
        val[..., 0] = 200*mu*(x-x**2)**2 * (2*y**3-3*y**2+y) * (2*z**3-3*z**2+z)
        val[..., 1] = -100*mu*(y-y**2)**2 * (2*x**3-3*x**2+x) * (2*z**3-3*z**2+z)
        val[..., 2] = -100*mu*(z-z**2)**2 * (2*y**3-3*y**2+y) * (2*x**3-3*x**2+x)

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)

parser = argparse.ArgumentParser(description="TetrahedronMesh 上的任意次 Lagrange 有限元空间的线性弹性问题求解.")
parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')
parser.add_argument('--degree', 
                    default=1, type=int, 
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

pde = BoxDomainPolyUnloaded3d()
args = parser.parse_args()

bm.set_backend(args.backend)
nx, ny, nz = args.nx, args.ny, args.nz
mesh = TetrahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()
p = args.degree

tmr = timer("FEM Solver")
next(tmr)

maxit = 4
errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$', 'boundary']
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 3))
    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D')
    tmr.send('material')
    
    integrator = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=tensor_space.p+3, method='fast_3d')
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly()
    tmr.send('fast stiffness assembly')

    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()
    tmr.send('source assembly')

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
    tmr.send('boundary')

    uh = tensor_space.function()
    K = K.tocsr()
    uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
    tmr.send('solve(cg)')

    tmr.send(None)

    u_exact = tensor_space.interpolate(pde.solution)
    # errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh - u_exact)**2 * (1 / NDof[i])))
    errorMatrix[0, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
    # errorMatrix[2, i] = bm.sqrt(bm.sum(bm.abs(uh[isDDof] - u_exact[isDDof])**2 * (1 / NDof[i])))

    # print("errorMatrix:", errorMatrix)

    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
# print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))