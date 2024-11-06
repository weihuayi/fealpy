#!/usr/bin/env python3
#

import numpy as np

from fealpy.pde.poisson_model_2d import CrackData, LShapeRSinData, CosCosData, KelloggData, SinSinData
from fealpy.vem import PoissonCVEMModel 
from fealpy.tools.show import showmultirate
from fealpy.mesh.simple_mesh_generator import triangle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def show_solution(axes, mesh, uh):
    cell = mesh.ds.cell
    cellLocation = mesh.ds.cellLocation
    point = mesh.point
    cd = np.hsplit(cell, cellLocation[1:-1])
    poly3d = [[(point[i,0], point[i, 1], uh[i]) for i in idx ] for idx in cd]
    collection = Poly3DCollection(poly3d, linewidths=1, alpha=0.2)
    axes.add_collection3d(collection)
    p0 = [[(point[i,0], point[i, 1], -1) for i in idx ] for idx in cd]
    c0 = Poly3DCollection(p0, linewidths=1, alpha=0.2)
    axes.add_collection3d(c0)

pde = CosCosData()
maxit = 4

h = 0.2
box = [0, 1, 0, 1] # [0, 1]^2 domain
mesh = triangle(box, h, meshtype='polygon')

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - \Pi^\\nabla u_h||_0$ with p=1',
             '$|| u - \Pi^\\nabla u_h||_0$ with p=2',
             '$|| u - \Pi^\\nabla u_h||_0$ with p=5',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=1',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=2',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=5',
             ]

ps = [1, 2, 5]

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    for j, p in enumerate(ps):
        vem = PoissonCVEMModel(pde, mesh, p, q=7)
        vem.solve()
        Ndof[i] = mesh.number_of_cells()
        errorMatrix[j, i] = vem.L2_error()
        errorMatrix[j+3, i] = vem.H1_semi_error()
    if i < maxit - 1:
        h /= 2
        mesh = triangle(box, h, meshtype='polygon')

print(errorMatrix)
mesh.add_plot(plt, cellcolor='w')
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()


