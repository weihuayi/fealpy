#!/usr/bin/env python3
# 
"""
Lagrange 元求解 非线性的 Poisson 方程, 

.. math::
    -\\nabla\cdot(a(u)\\nabla u) = f

Notes
-----

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.boundarycondition import NeumannBC
from fealpy.tools.show import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解 Poisson 方程
        边界条件为纯 Neumann 条件
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nspace',
        default=10, type=int,
        help='初始网格 x  和 y 方向剖分段数, 默认 10 段.')

parser.add_argument('--tol',
        default=1e-8, type=float,
        help='Picard 迭代的阈值, 默认 1e-8.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
nspace = args.nspace
maxit = args.maxit
tol = args.tol


# 定义模型数据

@cartesian
def solution(p):
    # 真解函数
    pi = np.pi
    x = p[..., 0]
    y = p[..., 1]
    return np.cos(pi*x)*np.cos(pi*y)

@cartesian
def gradient(p):
    # 真解函数的梯度
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    val = np.zeros(p.shape, dtype=np.float64)
    val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
    val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
    return val #val.shape ==p.shape

@cartesian
def source(p):
    # 源项
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    val = 2*pi**2*(3*np.cos(pi*x)**2*np.cos(pi*y)**2-np.cos(pi*x)**2-np.cos(pi*y)**2+1)*np.cos(pi*x)*np.cos(pi*y)
    return val

@cartesian
def dirichlet(p):
    return solution(p)

@cartesian
def neumann(p,n):
    grad = gradient(p) # (NQ, NE, 2)
    val = np.sum(grad*n, axis=-1) # (NQ, NE)
    return val

@cartesian
def is_dirichlet_boundary(p):
    y = p[..., 1]
    return (y == 1.0) | (y == 0.0)

@cartesian
def is_neumann_boundary(p):
    x = p[..., 0]
    return (x == 1.0) | (x == 0.0)


domain =[0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=nspace, ny=nspace, meshtype='tri')

NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

for i in range(maxit):
    print(i, ":")
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    b = space.source_vector(source)  
    space.set_neumann_bc(neumann, b, threshold = is_neumann_boundary)
    uh0 = space.function() # uh0 是当前迭代步的数值解
    uh1 = space.function() # uh1 是下一迭代步的数值解
    
    space.set_dirichlet_bc(dirichlet, uh0, threshold = is_dirichlet_boundary)
    
    bc = DirichletBC(space, dirichlet, threshold=is_dirichlet_boundary)

    @barycentric
    def dcoefficient(bcs):
        return 1+uh0(bcs)**2

    # picard 非线性迭代
    k = 0
    while True:
        uh1[:] = 0
        A = space.stiff_matrix(c=dcoefficient)
        F = b.copy()
        A, F = bc.apply(A, F, uh1)
        uh1[:] = spsolve(A, F).reshape(-1)
        err = np.max(np.abs(uh1-uh0))
        print(k, "-th picard with error", err)
        if err < tol:
            break
        else:
            k += 1
            uh0[:] = uh1 

    errorMatrix[0, i] = space.integralalg.error(solution, uh1.value)
    errorMatrix[1, i] = space.integralalg.error(gradient, uh1.grad_value)
    if i < maxit-1:
        mesh.uniform_refine()    

print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()




