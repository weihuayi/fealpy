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
mesh = pde.init_mesh(n=4, meshtype='tri')

maxit = 30
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

mesh.add_plot(plt)
plt.savefig('/home/why/test-0.png')
plt.close()

for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, 1, q=3)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_nodes()
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    rguh = ralg.harmonic_average(uh)
    #eta = fem.recover_estimate(rguh)
    eta = fem.residual_estimate()
    errorMatrix[3, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        # isMarkedCell = mark(eta, theta=theta)
        options = mesh.adaptive_options(method='max', theta=0.5)
        mesh.adaptive(eta, options)
        mesh.add_plot(plt)
        plt.savefig('/home/why/test-' + str(i+1) + '.png')
        plt.close()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, k, Ndof, errorMatrix, errorType)
plt.show()
