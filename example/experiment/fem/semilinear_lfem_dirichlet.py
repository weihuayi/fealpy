
import ipdb
import argparse
from matplotlib import pyplot as plt

from fealpy.experimental import logger
logger.setLevel('WARNING')

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import NonlinearForm
from fealpy.experimental.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.experimental.fem import ScalarSourceIntegrator
from fealpy.experimental.solver import cg

from fealpy.utils import timer


import numpy as np
from fealpy.decorator import cartesian
from typing import Sequence


PI = np.pi

class NonlinearData():
    def __init__(self, domain:Sequence[float]):

        self.domain = domain

    #扩散项系数
    @cartesian
    def diffusion_coefficient(self, p):

        A_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        A_coe[:] = 10 
        return A_coe

    @cartesian
    def diffusion_coefficient_right(self, p):

        return -self.diffusion_coefficient(p)

    #反应项系数
    @cartesian
    def reaction_coefficient(self, p):

        B_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        B_coe = 1
        return B_coe

    @cartesian
    def reaction_coefficient_right(self, p):

        return -self.reaction_coefficient(p)

    #真解
    @cartesian
    def solution(self, p):

        x = p[..., 0]
        y = p[..., 1]
        sol = np.zeros(p.shape[:-1], dtype=np.float64)
        sol[:] = np.sin(PI * x) * np.sin(PI * y)
        return sol

    #真解的梯度
    @cartesian
    def gradient(self, p):

        x = p[..., 0]
        y = p[..., 1]
        grad = np.zeros(p.shape, dtype = np.float64)
        grad[..., 0] = PI * np.cos(PI * x) * np.sin(PI * y)
        grad[..., 1] = PI * np.sin(PI * x) * np.cos(PI * y)
        return grad

    #源项
    @cartesian
    def source(self, p):

        sol = self.solution(p)
        a = self.diffusion_coefficient(p)
        b = self.reaction_coefficient(p)
        f = np.zeros(p.shape[:-1], dtype = np.float64)
        f[:] = 2 * a * PI**2 * sol + b * sol**3
        return f

    #边界条件
    @cartesian                       
    def dirichlet(self,p):

        return self.solution(p)

#非线性项函数以及导函数形式
def nonlinear_func(u):

    val = u**3
    return val

def nonlinear_gradient_func(u):

    val = 3*u**2
    return val

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
pde = NonlinearData(domain)
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)

p = 1
maxit = 6
tol = 1e-8
NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

#非线性迭代
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    u0 = space.function()
    du = space.function()
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)
    # isDDof = space.is_boundary_dof(pde.dirichlet, u0)
    isIDof = ~isDDof

    D = ScalarDiffusionIntegrator(u0, pde.diffusion_coefficient, q=p+2)
    M = ScalarMassIntegrator(u0, nonlinear_func, nonlinear_gradient_func, pde.reaction_coefficient, q=p+2)
    f = ScalarSourceIntegrator(pde.source, q=p+2)

#     while True:
#         n = NonlinearForm(space)
#         n.add_domain_integrator([D, M, f])
        
#         A, F = n.assembly()

#         du[isIDof] = spsolve(A[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
#         u0 += du
#         err = np.max(np.abs(du))

#         if err < tol:
#             break

#     uI = space.interpolate(pde.solution)
#     errorMatrix[0, i] = mesh.error(pde.solution, u0, q=p+2)
#     errorMatrix[1, i] = mesh.error(pde.gradient, u0.grad_value, q=p+2)
#     if i < maxit-1:
#         mesh.uniform_refine()

# #收敛阶    
# print(errorMatrix)
# print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
# showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20)   

# #可视化
# bc = np.array([1/3, 1/3, 1/3])
# uI = space.interpolate(pde.solution)
# uI = uI(bc)
# uh = u0(bc)

# fig, axes = plt.subplots(1, 2)
# mesh.add_plot(axes[0], cellcolor=uI, linewidths=0)
# mesh.add_plot(axes[1], cellcolor=uh, linewidths=0) 

# plt.show()
