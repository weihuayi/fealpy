
import numpy as np
import sys

from fealpy.model.obstacle_model_2d import ObstacleData1, ObstacleData2 
from fealpy.vemmodel.ObstacleVEMModel2d import ObstacleVEMModel2d
from fealpy.tools.show import showmultirate
from fealpy.quadrature import QuadrangleQuadrature 

import matplotlib.pyplot as plt

m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])

if m == 1:
    model = ObstacleData1()
    quadtree= model.init_mesh(n=3, meshtype='quadtree')
elif m == 2:
    model = ObstacleData2() 
    quadtree = model.init_mesh(n=4, meshtype='quadtree')

errorType = ['$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$'
             ]
mesh = quadtree.to_pmesh()
integrator = QuadrangleQuadrature(6)
vem = ObstacleVEMModel2d(model, mesh, p=p, integrator=integrator, quadtree=quadtree)

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    vem.solve()
    Ndof[i] = vem.V.number_of_global_dofs()
    errorMatrix[0, i] = vem.L2_error()
    errorMatrix[1, i] = vem.H1_semi_error()
    if i < maxit - 1:
        quadtree.uniform_refine()
        mesh = quadtree.to_pmesh()
        vem.reinit(mesh)

mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
s = axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)


fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca(projection='3d')
s = axes.plot_trisurf(x, y, tri, vem.uI-vem.gI, cmap=plt.cm.jet, lw=0.0)
fig3.colorbar(s)

showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
