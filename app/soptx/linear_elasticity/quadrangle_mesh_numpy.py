from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC

from fealpy.decorator import cartesian

from fealpy.solver import cg

from app.soptx.soptx.utilfs.timer import timer

from fealpy.material.elastic_material import LinearElasticMaterial


import argparse

# 平面应变问题
class BoxDomainData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=points.device)
        val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
        val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=points.device)
        val[..., 0] = x * (1 - x) * y * (1 - y)
        val[..., 1] = 0
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    

parser = argparse.ArgumentParser(description="QuadrangleMesh 上的任意次 Lagrange 有限元空间的线性弹性问题求解.")
parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')
parser.add_argument('--degree', 
                    default=2, type=int, 
                    help='Lagrange 有限元空间的次数, 默认为 1 次.')
parser.add_argument('--nx', 
                    default=4, type=int, 
                    help='x 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--ny',
                    default=4, type=int,
                    help='y 方向的初始网格单元数, 默认为 2.')
args = parser.parse_args()

bm.set_backend(args.backend)
pde = BoxDomainData2D()
nx, ny = args.nx, args.ny
extent = pde.domain()
mesh = QuadrangleMesh.from_box(box=extent, nx=nx, ny=ny, device='cpu')
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

p = args.degree

tmr = timer("FEM Solver")
next(tmr)

maxit = 5
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
    gdof = space.number_of_global_dofs()
    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='E1nu0.3', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_strain')
    tmr.send('material')
    
    integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=tensor_space.p+3)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')
    tmr.send('stiffness assembly')
    
    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()
    tmr.send('source assembly')

    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64)
    uh_bd, isDDof = tensor_space.boundary_interpolate(gD=pde.dirichlet, uh=uh_bd, threshold=None)

    F = F - K.matmul(uh_bd)
    F[isDDof] = uh_bd[isDDof]

    dbc = DirichletBC(space=tensor_space)
    K = dbc.apply_matrix(matrix=K, check=True)
    tmr.send('boundary')

    uh = tensor_space.function()

    uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
    tmr.send('solve(cg)')

    tmr.send(None)

    u_exact = tensor_space.interpolate(pde.solution)
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))
    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)

    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))