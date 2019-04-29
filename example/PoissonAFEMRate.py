#!/usr/bin/env python3
#

__doc__ = """

This is a scirpt for adaptive finite element method for Poisson example
"""

import numpy as np

import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate

pde = LShapeRSinData()
mesh = pde.init_mesh(n=2, meshtype='tri')

maxit = 50
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
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_error()
    rguh = ralg.harmonic_average(uh)
    eta = fem.recover_estimate(rguh)
    errorMatrix[3, i] = fem.get_recover_error(rguh)
    markedCell = mark(eta, theta=theta)
    if i < maxit - 1:
        isMarkedCell = mark(eta, theta=theta)
        mesh.bisect(isMarkedCell)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
