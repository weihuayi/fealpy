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
from fealpy.tools.show import showmultirate


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
    val[...,0] = -pi*np.sin(pi*x)*np.cos(pi*y)
    val[...,1] = -pi*np.cos(pi*x)*np.sin(pi*y)
    return val #val.shape ==p.shape

@cartesian
def source(p):
    # 源项
    x = p[...,0]
    y = p[...,1]
    pi = np.pi
    val = 2*pi**2*(3*np.cos(pi*x)**2*np.cos(pi*y)**2-np.cos(pi*x)**2-np.cos(pi*y)**2+1)*np.cos(pi*x)*np.cos(pi*y)
    return val

@cartesian
def dirichlet(p):
    return solution(p)


def nolinear_matrix(space, c, q=3):
    mesh = space.mesh
    qf = mesh.integrator(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    cellmeasure = mesh.entity_measure('cell')
    phi = space.basis(bcs)       # (NQ, 1, ldof)
    gphi = space.grad_basis(bcs) # (NQ, NC, ldof, GD)
    val = c(bcs)                 # (NQ, NC, GD)
    
    B = np.einsum('q, qcid, qcd, qcj, c->cij', ws, gphi, val, phi, cellmeasure)

    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:, :, None], shape=B.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)), shape=(gdof,gdof))

    return B

p = 1
maxit = 4 
tol = 1e-8

domain =[0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')

NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

for i in range(maxit):
    print(i, ":")
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    b = space.source_vector(source)
    uh = space.function() #  
    du = space.function() #
    isDDof = space.set_dirichlet_bc(uh, dirichlet)
    isIDof = ~isDDof

    @barycentric
    def dcoefficient(bcs):
        return 1+uh(bcs)**2

    @barycentric
    def nlcoefficient(bcs):
        return 2*uh(bcs)[...,None]*uh.grad_value(bcs)

    # 非线性迭代
    while True:
        A = space.stiff_matrix(c=dcoefficient)
        B = nolinear_matrix(space, nlcoefficient)
        U = A + B
        F = b - A@uh
        du[isIDof] = spsolve(U[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        uh += du
        err = np.max(np.abs(du))
        print(err)
        if err < tol:
           break

    errorMatrix[0, i] = space.integralalg.error(solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(gradient, uh.grad_value)
    if i < maxit-1:
        mesh.uniform_refine()
    
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()

