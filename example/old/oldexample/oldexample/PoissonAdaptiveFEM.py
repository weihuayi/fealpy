import sys
import numpy as np

from fealpy.model.poisson_model_2d import LShapeRSinData,CrackData,KelloggData,CosCosData,SinSinData

from fealpy.femmodel.PoissonAdaptiveFEMModel import PoissonAdaptiveFEMModel

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate
from fealpy.quadrature  import TriangleQuadrature

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])

if m == 1:
    pde = LShapeRSinData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 2:
    pde = KelloggData() #TODO
    mesh = pde.init_mesh(n=4,meshtype='tri')
elif m == 3:
    pde = CrackData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 4:
    pde = SinSinData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 5:
    pde = CosCosData()
    mesh = pde.init_mesh(n=4, meshtype='tri')


theta = 0.3

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$', 
             '$||\\nabla u - G(\\nabla u_h)||_{0}sim$',
]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

integrator = TriangleQuadrature(6)
fem = PoissonAdaptiveFEMModel(pde, mesh, p=p, integrator=integrator)

for i in range(maxit):
    print('step:', i)
    fem.solve()
    eta = fem.recover_estimate()
    Ndof[i] = fem.femspace.number_of_global_dofs()
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    errorMatrix[3, i] = np.sqrt(np.sum(eta**2))
    markedCell = mark(eta,theta=theta)  

    if i < maxit - 1:
        markedCell = mark(eta,theta=theta)
        mesh.bisect(markedCell)
        fem.reinit(mesh)   
mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
cell = mesh.ds.cell
axes.plot_trisurf(x, y, cell, fem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
showmultirate(plt, 2, Ndof, errorMatrix, errorType)
plt.show()

