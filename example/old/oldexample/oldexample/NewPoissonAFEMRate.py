#!/usr/bin/env python3
# 

import numpy as np
from fealpy.fem import PoissonFEMModel
from fealpy.tools.show import showmultirate
from fealpy.recovery import FEMFunctionRecoveryAlg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.pde.poisson_2d import LShapeRSinData

maxit = 15
p = 1
q = 5
k = maxit - 5

pde = LShapeRSinData()
tritree = pde.init_mesh(n=4, meshtype='tritree')
ralg = FEMFunctionRecoveryAlg()

errorType = [
        '$\| u - u_h\|_0$',
        '$\|\\nabla u - \\nabla u_h\|$',
        '$\eta$'
        ]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = tritree.to_conformmesh()

for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, p=p, q=q)
    fem.solve()
    Ndof[i] = fem.space.number_of_global_dofs()
    errorMatrix[0, i] = fem.L2_error()
    errorMatrix[1, i] = fem.H1_semi_error()
    rguh = ralg.harmonic_average(fem.uh)
    eta = fem.recover_estimate(rguh)
    errorMatrix[2, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        options = tritree.adaptive_options()
        tritree.adaptive(eta, options)
        mesh = tritree.to_conformmesh()

mesh.add_plot(plt, showaxis=True)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

showmultirate(plt, k, Ndof, errorMatrix, errorType)
plt.show()
