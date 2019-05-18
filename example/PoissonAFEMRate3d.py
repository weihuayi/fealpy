#!/usr/bin/env python3
#

""" Adaptive FEM for 3d Poisson PDE Model
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

from fealpy.pde.poisson_3d import LShapeRSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate

def savemesh(mesh, fname):
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    data = {'node': node, 'elem': cell+1}
    sio.matlab.savemat(fname, data)

def u(p):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    return x + y + z


pde = LShapeRSinData()
mesh = pde.init_mesh(n=4, meshtype='tet')

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

node = mesh.entity('node')
u0 = u(node)
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
    savemesh(mesh, '/home/why/result/test/test_' + str(i) + '.mat')
    if i < maxit - 1:
        isMarkedCell = mark(eta, theta=theta)
        if i == 2:
            print(i)
        A = mesh.bisect(isMarkedCell, returnim=True)
        node = mesh.entity('node')
        ui = u(node)
        u0 = A@u0
        print('Interpolation error: ', np.max(np.abs(ui - u0)))

fig = plt.figure()
axes = Axes3D(fig)
mesh.add_plot(axes, alpha=1)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
