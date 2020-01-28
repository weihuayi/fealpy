import numpy as np
import sys

from fealpy.pde.poisson_interface_model_2d import CircleInterfaceData, SquareInterfaceData
from fealpy.pde.poisson_interface_model_2d import FoldCurveInterfaceData
from fealpy.vem import PoissonInterfaceVEMModel
from fealpy.mesh.adaptive_tools import AdaptiveMarker
from fealpy.tools.show import showmultirate

from fealpy.mesh.implicit_curve import Circle
from fealpy.mesh.adaptive_interface_mesh_generator import AdaptiveMarker2d, QuadtreeInterfaceMesh2d 
from fealpy.quadrature import TriangleQuadrature 


import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])

if m == 1:
    model = CircleInterfaceData(np.array([0,0]), 0.53, 1, 1000)
    quadtree = model.init_mesh(n=1, meshtype='quadtree')
if m == 2:
    model = Circle1InterfaceData([0,0],2,1,1,0.001, 3)
    quadtree = model.init_mesh(n=4, meshtype='quadtree')
if m == 3:
    model = SquareInterfaceData([0,0],2,2,1)
    quadtree = model.init_mesh(n=4, meshtype='quadtree')
if m == 4:
    model = FoldCurveInterfaceData(6, 1, 1)
    quadtree = model.init_mesh(n=4, meshtype='quadtree')

errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla\Pi^\Delta u_h\|$',
             '$\|\\nabla\Pi^\Delta u_h -\Pi^\DeltaG(\\nabla\Pi^\Delta u_h)\|$'
            ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)


marker = AdaptiveMarker2d(model.interface, maxh=0.1, maxa=2)
alg = QuadtreeInterfaceMesh2d(quadtree, marker)
pmesh= alg.get_interface_mesh()

pmesh.add_plot(plt, cellcolor='w')
integrator = TriangleQuadrature(3)
vem = PoissonInterfaceVEMModel(model, pmesh, p=1, integrator=integrator)

for i in range(maxit):
    print('step:', i)
    vem.solve()
    Ndof[i] = vem.vemspace.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error() 
    errorMatrix[2, i] = vem.L2_error()
    errorMatrix[3, i] = vem.H1_semi_error()
    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        quadtree.uniform_refine()
        pmesh = alg.get_interface_mesh()
        vem.reinit(pmesh)

k,l = 0, 4
showmultirate(plt, k, Ndof, errorMatrix[:l, :], errorType[:l])
plt.show()

