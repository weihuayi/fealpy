import numpy as np

CONTEXT = 'numpy'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer
from fealpy.decorator import cartesian

from fealpy.np.mesh import TriangleMesh
from fealpy.torch.mesh import TriangleMesh as tri
from fealpy.np.functionspace import LagrangeFESpace
from fealpy.np.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt

from typing import Sequence

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


NX, NY = 64, 64
PI = np.pi
domain = [0, 1, 0, 2]
pde = NonlinearData(domain)
mesh = TriangleMesh.from_box(nx=NX, ny=NY)

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
    isIDof = ~isDDof

    D = ScalarDiffusionIntegrator(uh=u0, c=pde.diffusion_coefficient, q=p+2)
    M = ScalarMassIntegrator(uh=u0, uh_func=nonlinear_func, grad_uh_func=nonlinear_gradient_func, c=pde.reaction_coefficient, q=p+2)
    f = ScalarSourceIntegrator(pde.source, q=p+2)

    while True:
        n = NonlinearForm(space)
        n.add_domain_integrator([D, M, f])
        
        A, F = n.assembly()

        du[isIDof] = spsolve(A[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        u0 += du
        err = np.max(np.abs(du))

        if err < tol:
            break


