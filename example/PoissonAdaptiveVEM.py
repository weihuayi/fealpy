import numpy as np
import sys

from fealpy.model.poisson_model_2d import LShapeRSinData, CosCosData, KelloggData
from fealpy.vemmodel import PoissonVEMModel 
from fealpy.mesh.adaptive_tools import AdaptiveMarker 
from fealpy.tools.show import showmultirate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])

if m == 1:
    model = KelloggData()
    quadtree = model.init_mesh(n=4)
elif m == 2:
    model = LShapeRSinData() 
    quadtree = model.init_mesh(n=4)
elif m == 3:
    model = CosCosData()
    quadtree = model.init_mesh(n=4)

k = maxit - 10 
errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - u_h\|_0$',
             '$\|\\nabla \Pi u_h - \Pi G(\\nabla \Pi u_h) \|$',
             '$\|\\nabla u - \\nabla u_h\|$',
             '$\|\\nabla u - G(\\nabla u_h)\|$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = quadtree.to_pmesh()
vem = PoissonVEMModel(model, mesh, p=1)
for i in range(maxit):
    print('step:', i)
    vem.solve()
    eta = vem.recover_estimate()
    Ndof[i] = vem.V.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.interpolation_error()
    errorMatrix[2, i] = vem.L2_error(3, quadtree)
    errorMatrix[3, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        quadtree.refine(marker=AdaptiveMarker(eta, theta=0.2))
        vem.reinit(quadtree.to_pmesh())

mesh = vem.V.mesh

fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
mesh.add_plot(axes, cellcolor='w')
fig1.savefig('mesh.pdf')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh, cmap=plt.cm.jet, lw=0.0)
fig2.savefig('solution.pdf')


fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, k, Ndof, errorMatrix[:4, :], optionlist[:4], errorType[:4])
axes.legend(loc=3, prop={'size': 30})
plt.show()

