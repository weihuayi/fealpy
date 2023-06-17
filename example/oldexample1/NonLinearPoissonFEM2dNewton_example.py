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

from fealpy.decorator import cartesian, barycentric, timer
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate


# 定义模型数据

class CosCosData():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def solution(self, p):
        # 真解函数
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.cos(pi*x)*np.cos(pi*y)

    @cartesian
    def gradient(self, p):
        # 真解函数的梯度
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[...,0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[...,1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val #val.shape ==p.shape

    @cartesian
    def source(self, p):
        # 源项
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        val = 2*pi**2*(3*np.cos(pi*x)**2*np.cos(pi*y)**2-np.cos(pi*x)**2-np.cos(pi*y)**2+1)*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return x == 0.0

    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p)
        val = np.sum(grad*n, axis=-1)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        y = p[..., 1]
        return (y == 1.0) | (y == 0.0)

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

    @cartesian
    def is_robin_boundary(self, p):
        x = p[..., 0]
        return x == 1.0


@timer
def nolinear_matrix(uh, q=3):

    space = uh.space
    mesh = space.mesh

    qf = mesh.integrator(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    cellmeasure = mesh.entity_measure('cell')

    cval = 2*uh(bcs)[...,None]*uh.grad_value(bcs) # (NQ, NC, GD)
    phi = space.basis(bcs)       # (NQ, 1, ldof)
    gphi = space.grad_basis(bcs) # (NQ, NC, ldof, GD)

    B = np.einsum('q, qcid, qcd, qcj, c->cij', ws, gphi, cval, phi, cellmeasure)

    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:, :, None], shape=B.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)), shape=(gdof,gdof))

    return B

p = 2
maxit = 4 
tol = 1e-8

pde = CosCosData()

mesh = MF.boxmesh2d(pde.domain(), nx=10, ny=10, meshtype='tri')

NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

for i in range(maxit):
    print(i, ":")
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    u0 = space.function() #  
    du = space.function() #

    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0, threshold=pde.is_dirichlet_boundary)
    isIDof = ~isDDof

    b = space.source_vector(pde.source)

    b = space.set_neumann_bc(pde.neumann, b, threshold=pde.is_neumann_boundary)

    R, b = space.set_robin_bc(pde.robin, b, threshold=pde.is_robin_boundary)

    @barycentric
    def dcoefficient(bcs):
        return 1+u0(bcs)**2

    # 非线性迭代
    while True:
        A = space.stiff_matrix(c=dcoefficient)
        B = nolinear_matrix(u0, q=p+1)
        U = A + R + B
        F = b - A@u0 - R@u0
        du[isIDof] = spsolve(U[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        u0 += du
        err = np.max(np.abs(du))
        print(err)
        if err < tol:
           break

    errorMatrix[0, i] = space.integralalg.error(pde.solution, u0.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, u0.grad_value)
    if i < maxit-1:
        mesh.uniform_refine()
    
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()

