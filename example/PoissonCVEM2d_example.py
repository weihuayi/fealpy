#!/usr/bin/env python3
#

import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory, HalfEdgeMesh2d
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate


p = int(sys.argv[1])
n = int(sys.argv[2]) 
maxit = 4

pde = CosCosData()
box = pde.domain()

mf = MeshFactory()

errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
             '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='poly') 
    space = ConformingVirtualElementSpace2d(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    bc = DirichletBC(space, pde.dirichlet)
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)

    sh = space.project_to_smspace(uh)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, sh.value, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, sh.grad_value,
            power=2)
    n *= 2



mesh.add_plot(plt)
uh.add_plot(plt, cmap='rainbow')
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20, lw=2,
        ms=4)
plt.show()

if False:
    halfedge = mesh.entity('halfedge')
    hlevel = mesh.halfedgedata['level']
    NN = mesh.number_of_nodes()
    level = np.zeros(NN, dtype=np.int_)
    subdomain = mesh.ds.subdomain
    flag = subdomain[halfedge[:, 1]] > 0
    level[halfedge[flag, 0]] = hlevel[flag]
