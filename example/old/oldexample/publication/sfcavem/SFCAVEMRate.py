#!/usr/bin/env python3
#
"""
This script include the test code of the adaptive VEM for the simplified
friction problem.

Note
----
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from fealpy.pde.sfc_2d import SFCModelData1
from fealpy.vem.SFCVEMModel2d import SFCVEMModel2d
from fealpy.tools.show import showmultirate

import pickle

maxit = 43
theta = 0.2
k = maxit - 10

# prepare the pde model
pde = SFCModelData1()

qtree = pde.init_mesh(n=4, meshtype='quadtree')
mesh = qtree.to_pmesh()

errorType = ['$\eta$', '$\Psi_0$', '$\Psi_1$']
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
data = {}

for i in range(maxit):
    print(i, "-th step")
    vem = SFCVEMModel2d(pde, mesh, p=1, q=4)
    if i == 0:
        vem.solve(rho=0.7, maxit=40000)
    else:
        vem.solve(rho=0.9, maxit=40000, uh=data[2*(i-1)], lh=data[2*(i-1)+1])

    data[2*i] = vem.uh
    data[2*i+1] = vem.lh

    eta = vem.residual_estimator()
    psi0, psi1 = vem.high_order_term()
    Ndof[i] = vem.space.number_of_global_dofs()
    errorMatrix[0, i] = np.sqrt(np.sum(eta**2))
    errorMatrix[1, i] = np.sqrt(psi0)
    errorMatrix[2, i] = np.sqrt(psi1)

    # save data
    fname = sys.argv[1] + 'vem' + str(i) + '.space'
    f = open(fname, 'wb')
    pickle.dump([vem, qtree, data], f)

    fname = sys.argv[1] + 'error' + '-' + str(i) + '.data'
    f = open(fname, 'wb')
    pickle.dump([Ndof[:i+1], errorMatrix[:, :i+1], errorType], f)

    node = mesh.entity('node')
    x = node[:, 0]
    y = node[:, 1]
    tri = qtree.leaf_cell(celltype='tri')

    fig0 = plt.figure()
    fig0.set_facecolor('white')
    axes = fig0.gca(projection='3d')
    axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
    plt.savefig(sys.argv[1] + str(i) + '-solution.png')
    plt.close()

    fig1 = plt.figure()
    axes = fig1.gca()
    mesh.add_plot(axes)
    plt.savefig(sys.argv[1] + str(i) + '-mesh.png')
    plt.close()

    if i < maxit - 1:
        isMarkedCell = qtree.refine_marker(eta, theta, method="L2")
        qtree.refine(isMarkedCell, data=data)
        mesh = qtree.to_pmesh()

fname = sys.argv[1] + 'error.data'
f = open(fname, 'wb')
pickle.dump([Ndof, errorMatrix, errorType], f)
showmultirate(plt, k, Ndof, errorMatrix, errorType)
plt.show()
