import ipdb
import argparse
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.utils import timer
from fealpy.experimental import logger
logger.setLevel('WARNING')
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.pde.surface_poisson_model import SurfaceLevelSetPDEData
from fealpy.geometry import SphereSurface
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.experimental.functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from fealpy.experimental.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.experimental.fem import LinearForm, ScalarSourceIntegrator
from fealpy.experimental.sparse import COOTensor
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.experimental.solver import cg
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
        default=3, type=int,
        help='网格的阶数, 默认为 3 次.')

parser.add_argument('--mtype',
        default='tri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

args = parser.parse_args()

bm.set_backend(args.backend)
sdegree = args.sdegree
mdegree = args.mdegree
mtype = args.mtype

#tmr = timer()
#next(tmr)
x, y, z = sp.symbols('x, y, z', real=True)
F = x**2 + y**2 + z**2
u = x * y
pde = SurfaceLevelSetPDEData(F, u)

p = mdegree
surface = SphereSurface()
tmesh = TriangleMesh.from_unit_sphere_surface()
mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh, p, surface=surface)
#fname = f"sphere_test.vtu"
#mesh.to_vtk(fname=fname)

space = ParametricLagrangeFESpace(mesh, p=sdegree)
#tmr.send(f'第{i}次空间时间')
uI = space.interpolate(pde.solution)
uh = space.function()

bfrom = BilinearForm(space)
bfrom.add_integrator(ScalarDiffusionIntegrator())
lfrom = LinearForm(space)
lfrom.add_integrator(ScalarSourceIntegrator(pde.source))

A = bfrom.assembly()
F = lfrom.assembly()
#tmr.send(f'第{i}次矩阵组装时间')

C = space.integral_basis()
A = COOTensor.concat()
#A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
F = bm.r_[F, 0]

uh[:] = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14)

error = mesh.error(pde.solution, uh)
print(error)
