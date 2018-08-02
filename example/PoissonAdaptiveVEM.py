import numpy as np
import sys

from fealpy.pde.poisson_model_2d import CrackData, LShapeRSinData, CosCosData, KelloggData, SinSinData, ffData
from fealpy.vem import PoissonVEMModel 
from fealpy.mesh.adaptive_tools import AdaptiveMarker 
from fealpy.tools.show import showmultirate
from fealpy.quadrature import TriangleQuadrature 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])

if m == 1:
    pde = KelloggData()
    quadtree = pde.init_mesh(n=4)
elif m == 2:
    pde = LShapeRSinData() 
    quadtree = pde.init_mesh(n=4)
elif m == 3:
    pde = CrackData()
    quadtree = pde.init_mesh(n=4)
elif m == 4:
    pde = CosCosData()
    quadtree = pde.init_mesh(n=2)
elif m == 5:
    pde = SinSinData()
    quadtree = pde.init_mesh(n=3)
elif m == 6:
    pde = ffData()
    quadtree = pde.init_mesh(n=2)




theta = 0.2

k = maxit - 15 
errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
             '$\|\\nabla \Pi^\Delta u_h - \Pi^\Delta G(\\nabla \Pi^\Delta u_h) \|$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = quadtree.to_pmesh()

integrator = TriangleQuadrature(6)
for i in range(maxit):
    print('step:', i)
    vem = PoissonVEMModel(pde, mesh, p=p, integrator=integrator)
    vem.solve()
    eta = vem.recover_estimate(residual=True)
    Ndof[i] = vem.vemspace.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error() 
    errorMatrix[2, i] = vem.L2_error()
    errorMatrix[3, i] = vem.H1_semi_error()
    errorMatrix[0, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        quadtree.refine(marker=AdaptiveMarker(eta, theta=theta))
        mesh = quadtree.to_pmesh()

mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)

showmultirate(plt, k, Ndof, errorMatrix, errorType)
plt.show()

