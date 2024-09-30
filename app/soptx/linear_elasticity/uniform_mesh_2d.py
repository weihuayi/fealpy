from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.decorator import cartesian
from fealpy.experimental.mesh import UniformMesh2d, QuadrangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem.linear_form import LinearForm
from fealpy.experimental.decorator import cartesian

from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg

from fealpy.utils import timer

import argparse

# 平面应变问题
class BoxDomainData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
        val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = x * (1 - x) * y * (1 - y)
        val[..., 1] = 0
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    

parser = argparse.ArgumentParser(description="UniformMesh2d 上的任意次 Lagrange 有限元空间的线性弹性问题求解.")
parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')
parser.add_argument('--degree', 
                    default=2, type=int, 
                    help='Lagrange 有限元空间的次数, 默认为 1 次.')
parser.add_argument('--nx', 
                    default=2, type=int, 
                    help='x 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--ny',
                    default=2, type=int,
                    help='y 方向的初始网格单元数, 默认为 2.')
args = parser.parse_args()

pde = BoxDomainData2D()
args = parser.parse_args()

bm.set_backend(args.backend)
nx, ny = args.nx, args.ny
extent = pde.domain()
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]
mesh = UniformMesh2d(extent=[0, 1, 0, 1], h=h, origin=origin, 
                    ipoints_ordering='nec')

# mesh = QuadrangleMesh.from_box(box=extent, nx=nx, ny=ny)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

p = args.degree

tmr = timer()

maxit = 4
errorMatrix = bm.zeros((3, maxit), dtype=bm.float64)
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$', 'boundary']
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
    gdof = space.number_of_global_dofs()
    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='E1nu0.3', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_strain')
    integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=tensor_space.p+3)
    next(tmr)
    KE = integrator_K.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    
    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    tmr.send('forms')

    K = bform.assembly()
    F = lform.assembly()
    tmr.send('assembly')

    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64)
    uh_bd, isDDof = tensor_space.boundary_interpolate(gD=pde.dirichlet, uh=uh_bd, threshold=None)

    F_test = F.round(4)
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
    tmr.send('dirichlet')

    uh = tensor_space.function()

    uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
    tmr.send('solve(cg)')

    u_exact = tensor_space.interpolate(pde.solution)
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh - u_exact)**2 * (1 / NDof[i])))
    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
    errorMatrix[2, i] = bm.sqrt(bm.sum(bm.abs(uh[isDDof] - u_exact[isDDof])**2 * (1 / NDof[i])))
    print("errorMatrix:\n", errorMatrix)

    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))