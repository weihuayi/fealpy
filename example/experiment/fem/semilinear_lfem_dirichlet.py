
import ipdb
import argparse

from fealpy.experimental import logger
logger.setLevel('WARNING')

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import SemilinearForm
from fealpy.experimental.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.experimental.fem import ScalarSourceIntegrator
from fealpy.experimental.pde.semilinear_2d import SemilinearData
from fealpy.experimental.solver import cg
from fealpy.utils import timer


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解possion方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--meshtype',
        default='tri', type=str,
        help='默认网格为三角形网格')

args = parser.parse_args()


bm.set_backend(args.backend)
p = args.degree
n = args.n
meshtype = args.meshtype
maxit = args.maxit

tmr = timer()
next(tmr)

domain = [0, 1, 0, 2]
nx = 4
ny = 4
pde = SemilinearData(domain)
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)

p = 1
maxit = 6
tol = 1e-8
NDof = bm.zeros(maxit, dtype=bm.int64)
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
def func(u):
    return u**3

def grad_func(u):
    return 3*u**2

def diffusion_coef(p):
    return pde.diffusion_coefficient(p)

def reaction_coef(p):
    return pde.reaction_coefficient(p)

reaction_coef.func = func
reaction_coef.grad_func = grad_func
    
#非线性迭代
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    u0 = space.function()
    du = space.function()
    diffusion_coef.uh = u0
    reaction_coef.uh = u0
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)
    isIDof = ~isDDof

    D = ScalarDiffusionIntegrator(diffusion_coef, q=p+2, method='semilinear')
    M = ScalarMassIntegrator(reaction_coef, q=p+2, method='semilinear')
    f = ScalarSourceIntegrator(pde.source, q=p+2)

    while True:
        n = SemilinearForm(space)
        n.add_integrator([M, D])
        n.add_integrator(f)
        A, F = n.assembly()

        du[isIDof] = cg(A[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        u0 += du
        err = bm.max(bm.abs(du))

        if err < tol:
            break

    uI = space.interpolate(pde.solution)
    errorMatrix[0, i] = mesh.error(pde.solution, u0, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, u0.grad_value, q=p+2)
    if i < maxit-1:
        print('yes')
        mesh.uniform_refine()

#收敛阶    
print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
# showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20)   

# #可视化
# bc = bm.array([1/3, 1/3, 1/3])
# uI = space.interpolate(pde.solution)
# uI = uI(bc)
# uh = u0(bc)

# fig, axes = plt.subplots(1, 2)
# mesh.add_plot(axes[0], cellcolor=uI, linewidths=0)
# mesh.add_plot(axes[1], cellcolor=uh, linewidths=0) 

# plt.show()
