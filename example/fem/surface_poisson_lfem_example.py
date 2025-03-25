import ipdb
import argparse
import sympy as sp
import matplotlib.pyplot as plt
from scipy.sparse import coo_array, csr_array, bmat
from mpl_toolkits.mplot3d import Axes3D

from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm

from fealpy.pde.surface_poisson_model import SurfaceLevelSetPDEData
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.mesh import LagrangeTriangleMesh, LagrangeQuadrangleMesh
from fealpy.functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.solver import cg, spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        曲面上的任意次等参有限元方法
        """)

parser.add_argument('--sdegree',
        default=1, type=int,
        help='Lagrange 参数有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mdegree',
        default=1, type=int,
        help='网格的阶数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='ltri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

parser.add_argument('--maxit',
        default=4, type=int,
        help="默认网格加密求解次数，默认加密求解4次.")

args = parser.parse_args()
bm.set_backend(args.backend)

sdegree = args.sdegree
mdegree = args.mdegree
mtype = args.mtype
maxit = args.maxit

if mtype == 'ltri':
    LinearMesh = TriangleMesh
    LagrangeMesh = LagrangeTriangleMesh
elif mtype == 'lquad':
    LinearMesh = QuadrangleMesh
    LagrangeMesh = LagrangeQuadrangleMesh

tmr = timer()
next(tmr)
x, y, z = sp.symbols('x, y, z', real=True)
F = x**2 + y**2 + z**2
u = x * y
pde = SurfaceLevelSetPDEData(F, u)

p = mdegree
surface = SphereSurface()

lmesh = LinearMesh.from_unit_sphere_surface()


errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.float64)
h = bm.zeros(maxit, dtype=bm.float64)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    
    if mtype == 'ltri':
        mesh = LagrangeMesh.from_triangle_mesh(lmesh, p=mdegree, surface=surface)
    elif mtype == 'lquad':
        mesh = LagrangeMesh.from_quadrangle_mesh(lmesh, p=mdegree, surface=surface)
    
    space = ParametricLagrangeFESpace(mesh, p=sdegree)
    NDof[i] = space.number_of_global_dofs()
    h[i] = mesh.entity_measure('edge').max()

    uI = space.interpolate(pde.solution)

    bfrom = BilinearForm(space)
    bfrom.add_integrator(ScalarDiffusionIntegrator(method='isopara'))
    lfrom = LinearForm(space)
    lfrom.add_integrator(ScalarSourceIntegrator(pde.source, method='isopara'))

    A = bfrom.assembly(format='coo')
    F = lfrom.assembly()
    C = space.integral_basis()
    
    def coo(A):
        data = A._values
        indices = A._indices
        return coo_array((data, indices), shape=A.shape)
    A = bmat([[coo(A), C.reshape(-1,1)], [C[None,:], None]], format='coo')
    A = COOTensor(bm.stack([A.row, A.col], axis=0), A.data, spshape=A.shape)

    F = bm.concatenate((F, bm.array([0])))
    
    uh = space.function()
    x = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14).reshape(-1)
    uh[:] = -x[:-1]
    tmr.send(f'第{i}次求解时间')

    errorMatrix[0, i] = mesh.error(pde.solution, uh.value, q=p+3)

    if i < maxit-1:
        lmesh.uniform_refine()
    tmr.send(f'第{i}次去查计算及网格加密时间')
next(tmr)
print("最终误差:", errorMatrix)
print("order:", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 1, h, errorMatrix,  errorType, propsize=20)
plt.show()
