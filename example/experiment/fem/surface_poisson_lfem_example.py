import ipdb
import argparse
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm

from fealpy.pde.surface_poisson_model import SurfaceLevelSetPDEData
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.functionspace import LagrangeFESpace

# solver
from fealpy.solver import cg, spsolve
from scipy.sparse import coo_array, csr_array, bmat
#from scipy.sparse.linalg import spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        曲面上的任意次等参有限元方法
        """)

parser.add_argument('--sdegree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mdegree',
        default=1, type=int,
        help='网格的阶数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='ltri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

parser.add_argument('--refine',
        default=0, type=int,
        help="默认不加密网格.")

args = parser.parse_args()

bm.set_backend(args.backend)
sdegree = args.sdegree
mdegree = args.mdegree
mtype = args.mtype
refine = args.refine

#tmr = timer()
#next(tmr)
x, y, z = sp.symbols('x, y, z', real=True)
F = x**2 + y**2 + z**2
u = x * y
pde = SurfaceLevelSetPDEData(F, u)

p = mdegree
surface = SphereSurface()
tmesh = TriangleMesh.from_unit_sphere_surface()
tmesh.uniform_refine(n=refine)

mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh, p, surface=surface)
#fname = f"sphere_test.vtu"
#mesh.to_vtk(fname=fname)

space = ParametricLagrangeFESpace(mesh, p=sdegree)
#tmr.send(f'第{i}次空间时间')
uI0 = space.function()
uI0[:] = space.interpolate(pde.solution)
error = mesh.error(pde.solution, uI0)
print("uI error:", error)



if 0 :
    uh = space.function()

    bfrom = BilinearForm(space)
    bfrom.add_integrator(ScalarDiffusionIntegrator())
    lfrom = LinearForm(space)
    lfrom.add_integrator(ScalarSourceIntegrator(pde.source))

    A = bfrom.assembly(format='coo')
    F = lfrom.assembly()
    #tmr.send(f'第{i}次矩阵组装时间')

    C = space.integral_basis()

    def coo(A):
        data = A._values
        indices = A._indices
        return coo_array((data, indices), shape=A.shape)
    A = bmat([[coo(A), C.reshape(-1,1)], [C, None]], format='coo')
    A = COOTensor(bm.stack([A.row, A.col], axis=0), A.data, spshape=A.shape)

    F = bm.concatenate((F, bm.array([0])))

    x = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14).reshape(-1)
    uh[:] = x[:-1]
    #uh[:] = spsolve(A, F, 'scipy')[:-1]

    error = mesh.error(pde.solution, uh)
    print(error)
