#!/usr/bin/env python3
#

""" Adaptive FEM for 3d Poisson PDE Model
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_3d import LShapeRSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate

pde = LShapeRSinData()
mesh = pde.init_mesh(n=2, meshtype='tet')

maxit = 40
theta = 0.2
k = maxit - 15
p = 1
q = 3
errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}$']

ralg = FEMFunctionRecoveryAlg()
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, 1, q=3)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_nodes()
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    # rguh = ralg.harmonic_average(uh)
    # eta = fem.recover_estimate(rguh)
    eta = fem.residual_estimate()
    errorMatrix[3, i] = np.sqrt(np.sum(eta**2))
    markedCell = mark(eta, theta=theta)
    if i < maxit - 1:
        isMarkedCell = mark(eta, theta=theta)
        A = mesh.bisect(isMarkedCell, returnim=True)

fig = plt.figure()
axes = Axes3D(fig)
mesh.add_plot(axes, alpha=1)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
