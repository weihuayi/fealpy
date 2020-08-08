#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import IsoLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from fealpy.solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)

    space0 = IsoLagrangeFiniteElementSpace(lmesh, p=1)
    A0 = space0.stiff_matrix()

    space = IsoLagrangeFiniteElementSpace(lmesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.solution) 

    uh = space.function()
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    # 封闭曲面，设置其第 0 个插值节点的值为真解值
    A, F = bc.apply(A, F, uh, threshold=np.array([0], dtype=np.int_))
    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()


lmesh.nodedata['uh'] = uh
lmesh.to_vtk(fname='surface_with_solution.vtu')
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()
