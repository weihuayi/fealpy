import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.tools import showmultirate

#三角形网格
from fealpy.mesh import TriangleMesh

# 拉格朗日有限元空间
from fealpy.functionspace import LagrangeFESpace

#区域积分子
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator

#界面积分子
from fealpy.fem import ScalarInterfaceIntegrator 

#双线性形
from fealpy.fem import NonlinearForm

import numpy as np
from fealpy.decorator import cartesian
from typing import Sequence


class LineInterfaceData():
    def __init__(self, a:Sequence[float], b:Sequence[float], domain:Sequence[float]):

        self.a0 = a[0]
        self.a1 = a[1]
        self.b0 = b[0]
        self.b1 = b[1]
        self.domain = domain
    def interface_position(self, p):

        return p[..., 1] - 1

    def is_interface(self, p):

        y = p[..., 1]
        return np.abs(y - 1.0) < 1e-12

    #每个子区域对应的单元全局编号的布尔值
    @cartesian
    def subdomain(self, p):

        sdflag = [self.interface_position(p) < 0, self.interface_position(p) > 0]
        return sdflag

    #扩散项系数
    @cartesian
    def diffusion_coefficient(self, p):

        flag = self.subdomain(p) 
        A_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        A_coe[flag[0]] = self.a0
        A_coe[flag[1]] = self.a1
        return A_coe

    #反应项系数
    @cartesian
    def reaction_coefficient(self, p):

        flag = self.subdomain(p)
        B_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        B_coe[flag[0]] = self.b0
        B_coe[flag[1]] = self.b1
        return B_coe

    #真解
    @cartesian
    def solution(self, p):

        flag = self.subdomain(p)
        pi = np.pi
        sol = np.zeros(p.shape[:-1], dtype=np.float64)
        x = p[flag[0]][..., 0]
        y = p[flag[0]][..., 1]
        sol[flag[0]] = np.sin(pi*x)*np.sin(pi*y)
        x = p[flag[1]][..., 0]
        y = p[flag[1]][..., 1]
        sol[flag[1]] = -np.sin(pi*x)*np.sin(pi*y)
        return sol

    #真解的梯度
    @cartesian
    def gradient(self, p):

        flag = self.subdomain(p)
        pi = np.pi
        grad = np.zeros(p.shape, dtype = np.float64)
        x = p[...,flag[0], 0]
        y = p[...,flag[0], 1]
        grad[flag[0], 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad[flag[0], 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        x = p[...,flag[1], 0]
        y = p[...,flag[1], 1]
        grad[flag[1], 0] = -pi*np.cos(pi*x)*np.sin(pi*y)
        grad[flag[1], 1] = -pi*np.sin(pi*x)*np.cos(pi*y)
        return grad

    #源项
    @cartesian
    def source(self, p):

        flag = self.subdomain(p)
        pi = np.pi
        a0 = self.a0
        a1 = self.a1
        b0 = self.b0
        b1 = self.b1
        sol = self.solution(p)
        #f0 = 2*a0*pi^2*u0 + b0*u0^3
        #f1 = 2*a1*pi^2*u1 + b1*u1^3
        b = np.zeros(p.shape[:-1], dtype = np.float64)
        b[flag[0]] = 2*a0*pi**2*sol[flag[0]]+b0*sol[flag[0]]**3
        b[flag[1]] = 2*a1*pi**2*sol[flag[1]]+b1*sol[flag[1]]**3
        return b

    #边界条件
    @cartesian
    def neumann(self, p):

        flag = self.subdomain(p)
        a0 = self.a0
        a1 = self.a1
        grad = self.gradient(p)
        n = self.normal(p)
        neu = np.zeros(p.shape[:,-1], dtype = np.float64)
        neu[flag[0]] = a0*sum(grad[flag[0], :] * n[flag[0], :],axis = -1)
        neu[flag[1]] = a1*sum(grad[flag[1], :] * n[flag[1], :],axis = -1)
        return neu

    @cartesian                       
    def dirichlet(self,p):

        return self.solution(p)

    @cartesian
    def interfaceFun(self, p):

        a0 = self.a0
        a1 = self.a1
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        #gI = a0*grad[flag[0],:]*n[flag[0], :]+a1*grad[flag[1].:]*n[flag[1],:]
        #为方便，将方程中界面函数 gI 前面的负号移动到界面函数中
        grad0 = np.zeros(p.shape, dtype = np.float64)
        grad1 = np.zeros(p.shape, dtype = np.float64)
        grad0[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad0[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        grad1[..., 0] = -pi*np.cos(pi*x)*np.sin(pi*y)
        grad1[..., 1] = -pi*np.sin(pi*x)*np.cos(pi*y)
        n0 = np.zeros(p.shape, dtype = np.float64)
        n1 = np.zeros(p.shape, dtype = np.float64)
        n0[..., 0] = 0
        n0[..., 1] = 1
        n1[..., 0] = 0
        n1[..., 1] = -1
        gI = np.zeros(p.shape[:-1], dtype = np.float64)
        gI = a0 * np.sum(grad0 * n0, axis=-1) + a1 * np.sum(grad1 * n1, axis=-1)
        return gI

#寻找界面边索引
def interface_edge_index(mesh, pde):

    node = mesh.entity("node")
    edge = mesh.entity("edge")
    interface = pde.interface_position
    EdgeMidnode = 1/2 * (node[edge[:,0],:] + node[edge[:,1],:])
    isInterfaceEdge = (interface(EdgeMidnode) == 0)
    InterfaceEdgeIdx = np.nonzero(isInterfaceEdge)
    InterfaceEdgeIdx = InterfaceEdgeIdx[0]
    return InterfaceEdgeIdx
#非线性项函数以及导函数形式
def nonlinear_func(u):

    val = u**3
    return val

def nonlinear_gradient_func(u):

    val = 3*u**2
    return val

domain = [0, 1, 0, 2]
a = [10, 1]
b = [1, 0]
nx = 4
ny = 4
pde = LineInterfaceData(a, b, domain)
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)

p = 1
maxit = 6
tol = 1e-10
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
    I = ScalarInterfaceIntegrator(gI=pde.interfaceFun,  threshold=interface_edge_index(mesh, pde) , q=p+2)

    while True:
        n = NonlinearForm(space)
        n.add_domain_integrator([D, M, f])
        n.add_boundary_integrator([I])
        A, F = n.assembly()

        du[isIDof] = spsolve(A[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        u0 += du
        err = np.max(np.abs(du))

        if err < tol:
            break

    uI = space.interpolate(pde.solution)
    errorMatrix[0, i] = mesh.error(pde.solution, u0, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, u0.grad_value, q=p+2)
    if i < maxit-1:
        mesh.uniform_refine()

#收敛阶    
print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20)   

#真解
bc = np.array([1/3, 1/3, 1/3])
uI = space.interpolate(pde.solution)
uI = uI(bc)
uh = u0(bc)

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=uI, linewidths=0)
mesh.add_plot(axes[1], cellcolor=uh, linewidths=0) 

plt.show()
