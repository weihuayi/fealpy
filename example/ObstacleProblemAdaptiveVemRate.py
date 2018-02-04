
import numpy as np
import sys

from fealpy.model.obstacle_model_2d import ObstacleData1, ObstacleData2 
from fealpy.vemmodel.ObstacleVEMModel2d import ObstacleVEMModel2d
from fealpy.tools.show import showmultirate
from fealpy.quadrature import TriangleQuadrature 

from fealpy.mesh.adaptive_tools import AdaptiveMarker 

import matplotlib.pyplot as plt

m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])
theta = float(sys.argv[4])

if m == 1:
    model = ObstacleData1()
    quadtree= model.init_mesh(n=3, meshtype='quadtree')
elif m == 2:
    model = ObstacleData2() 
    quadtree = model.init_mesh(n=3, meshtype='quadtree')


k = maxit - 10
errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
             '$\|\\nabla \Pi^\Delta u_h - \Pi^\Delta G(\\nabla \Pi^\Delta u_h) \|$'
             ]
mesh = quadtree.to_pmesh()
integrator = TriangleQuadrature(3)
vem = ObstacleVEMModel2d(model, mesh, p=p, integrator=integrator)

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    vem.solve()
    eta = vem.recover_estimate()
    Ndof[i] = vem.vemspace.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error() 
    errorMatrix[2, i] = vem.L2_error()
    errorMatrix[3, i] = vem.H1_semi_error()
    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        #quadtree.uniform_refine()
        quadtree.refine(marker=AdaptiveMarker(eta, theta=theta))
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
