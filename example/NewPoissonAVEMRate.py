#!/usr/bin/env python3
# 

import numpy as np
from fealpy.vem import PoissonVEMModel
from fealpy.tools.show import showmultirate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = 2
p = 1
q = 3
maxit = 10
k = maxit - 5

if m == 1:
    from fealpy.pde.poisson_2d import KelloggData
    pde = KelloggData()
    quadtree = pde.init_mesh(n=4)
elif m == 2:
    from fealpy.pde.poisson_model_2d import LShapeRSinData
    pde = LShapeRSinData()
    quadtree = pde.init_mesh(n=4)
elif m == 3:
    from fealpy.pde.poisson_model_2d import CrackData
    pde = CrackData()
    quadtree = pde.init_mesh(n=4)
elif m == 4:
    from fealpy.pde.poisson_model_2d import CosCosData
    pde = CosCosData()
    quadtree = pde.init_mesh(n=2)
elif m == 5:
    from fealpy.pde.poisson_model_2d import SinSinData
    pde = SinSinData()
    quadtree = pde.init_mesh(n=3)
elif m == 6:
    from fealpy.pde.poisson_model_2d import ffData
    pde = ffData()
    quadtree = pde.init_mesh(n=2)

errorType = [
        '$\| u_I - u_h \|_{l_2}$',
        '$\|\\nabla u_I - \\nabla u_h\|_A$',
        '$\| u - \Pi^\Delta u_h\|_0$',
        '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
        '$\eta$'
        ]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = quadtree.to_pmesh()

for i in range(maxit):
    print('step:', i)
    vem = PoissonVEMModel(pde, mesh, p=p, q=q)
    vem.solve()
    eta = vem.recover_estimate(rtype='inv_area', residual=True)
    Ndof[i] = vem.space.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error()
    errorMatrix[2, i] = vem.L2_error()
    if m == 1:
        errorMatrix[3, i] = vem.H1_semi_error_Kellogg()
    else:
        errorMatrix[3, i] = vem.H1_semi_error()

    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        options = quadtree.adaptive_options()
        quadtree.adaptive(eta, options)
        mesh = quadtree.to_pmesh()

mesh.add_plot(plt, showaxis=True)

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)

showmultirate(plt, k, Ndof, errorMatrix[2:, :], errorType[2:])
plt.show()
