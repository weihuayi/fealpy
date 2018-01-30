import numpy as np
import sys

from fealpy.model.poisson_interface_model_2d import CircleInterfaceDataTest, Circle1InterfaceData, SquareInterfaceData
from fealpy.vemmodel import PoissonInterfaceVEMModel
from fealpy.mesh.adaptive_tools import AdaptiveMarker
from fealpy.tools.show import showmultirate

from fealpy.mesh.implicit_curve import Circle
from fealpy.mesh.adaptive_interface_mesh_generator import AdaptiveMarker2d, QuadtreeInterfaceMesh2d 

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])

if m == 1:
    model = CircleInterfaceDataTest(np.array([0,0]), 0.5, 1, 1)
    quadtree = model.init_mesh(n=3, meshtype='quadtree')
if m == 2:
    model = Circle1InterfaceData([0,0],2,1,1,0.001, 3)
    quadtree = model.init_mesh(n=4, meshtype='quadtree')
if m == 3:
    model = SquareInterfaceData([0,0],2,2,1)
    quadtree = model.init_mesh(n=4, meshtype='quadtree')

k = 0
errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla\Pi^\Delta u_h\|$',
             '$\|\\nabla\Pi^\Delta u_h -\Pi^\DeltaG(\\nabla\Pi^\Delta u_h)\|$'
            ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)


#marker = AdaptiveMarker2d(model.interface, maxh=0.1, maxa=2)
#alg = QuadtreeInterfaceMesh2d(quadtree, marker)
#pmesh= alg.get_interface_mesh()
#
#fig0 = plt.figure()
#fig0.set_facecolor('white')
#axes = fig0.gca() 
#pmesh.add_plot(axes, cellcolor='w')

vem = PoissonInterfaceVEMModel(model, quadtree.to_pmesh(), p=1)

for i in range(maxit):
    print('step:', i)
    vem.solve()
    print(vem.uh)
    Ndof[i] = vem.V.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
#    errorMatrix[1, i] = vem.uIuh_error() 
#    errorMatrix[2, i] = vem.L2_error(quadtree)
#    errorMatrix[3, i] = vem.H1_semi_error(quadtree)
#    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        quadtree.uniform_refine()
        #pmesh = alg.get_interface_mesh()
        pmesh = quadtree.to_pmesh()
        vem.reinit(pmesh)

fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
pmesh.add_plot(axes, cellcolor='w')


fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, k, Ndof, errorMatrix[:1, :], optionlist[:1], errorType[:1])
axes.legend(loc=3, prop={'size': 30})
plt.show()

