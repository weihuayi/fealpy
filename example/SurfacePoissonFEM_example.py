#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import IsoLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

from fealpy.solver import MatlabSolver
import transplant

# solver
from fealpy.solver import PETScSolver
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import pyamg

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             '$|| u - u_I||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u_I - u_h ||_{\Omega, \infty}$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

m = 4

for i in range(maxit):
    print("The {}-th computation:".format(i))

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
    space = IsoLagrangeFiniteElementSpace(lmesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    uh = space.function()
    A = space.surface_stiff_matrix()
    M = space.mass_matrix()

    if m == 1:
        C = space.integral_basis()
        barf = space.integralalg.mesh_integral(pde.source)/np.sum(space.cellmeasure)

        @cartesian
        def f(p):
            return pde.source(p) - barf 
        F = space.source_vector(f)

        A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
        F = np.r_[F, 0]

        x = spsolve(A, F).reshape(-1)
        uh[:] = x[:-1]
        baru = space.integralalg.mesh_integral(pde.solution)/np.sum(space.cellmeasure)
        uh += baru
    elif m == 2:
        F = space.source_vector(pde.source)
        A[0, 0] += 1e-8
        uh[:] = spsolve(A, F).reshape(-1)
    elif m == 3:
        @cartesian
        def f(p):
            return pde.solution(p) + pde.source(p)
        A += M
        F = space.source_vector(f)
        F -= np.mean(F)
        uh[:] = spsolve(A, F).reshape(-1)
        e = space.integralalg.mesh_integral(uh.value)/np.sum(space.cellmeasure)
        u = space.integralalg.mesh_integral(pde.solution)/np.sum(space.cellmeasure)
        uh += u - e
        print(e)
    elif m == 4:

        @barycentric
        def f(bc):
            p = lmesh.bc_to_point(bc)
            n = lmesh.cell_unit_normal(bc)
            return pde.source(p, n) 
        F = space.source_vector(f)

        C = space.integral_basis()
        A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
        F = np.r_[F, 0]

        x = spsolve(A, F).reshape(-1)
        uh[:] = x[:-1]

    @barycentric
    def gradient0(bc):
        p = lmesh.bc_to_point(bc)
        n = lmesh.cell_unit_normal(bc)
        val = pde.gradient(p, n)
        return val

    @barycentric
    def gradient1(bc):
        p0 = lmesh.bc_to_point(bc)
        n0 = lmesh.cell_unit_normal(bc)

        p1, d = surface.project(p0)
        n1 = surface.unit_normal(p1)
        H = surface.hessian(p0)

        val = uh.grad_value(bc)

        H *= -d[..., None, None] 
        H[..., [0, 1, 2], [0, 1, 2]] += 1
        H = np.linalg.inv(H)

        c = np.sum(n0*n1, axis=-1)[..., None]
        val -= np.sum(n1*val, axis=-1, keepdims=True)*n0/c
        val = np.einsum('...mn, ...n->...m', H, val)
        return val


    uI = space.interpolation(pde.solution)
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[2, i] = space.integralalg.error(pde.solution, uI.value)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uI.grad_value)
    errorMatrix[4, i] = np.max(np.abs(uI - uh))

    if i < maxit-1:
        mesh.uniform_refine()

lmesh.nodedata['uh'] = uh
lmesh.nodedata['uI'] = uI 

lmesh.to_vtk(fname='surface_with_solution.vtu')
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()
