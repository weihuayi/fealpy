import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_model_2d import LShapeRSinData, CrackData, KelloggData, CosCosData, SinSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh.adaptive_tools import mark
from fealpy.quadrature  import TriangleQuadrature

from mpl_toolkits.mplot3d import Axes3D
from fealpy.tools.show import showmultirate


m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])

if m == 1:
    pde = LShapeRSinData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 2:
    pde = KelloggData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 3:
    pde = CrackData()
    mesh = pde.init_mesh(n=4, meshtype='tri')
elif m == 4:
    pde = SinSinData()
    mesh = pde.init_mesh(n=1, meshtype='tri')
elif m == 5:
    pde = CosCosData()
    mesh = pde.init_mesh(n=2, meshtype='tri')


theta = 0.1
k = maxit - 15
errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}sim$']

ralg = FEMFunctionRecoveryAlg()
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
integrator = mesh.integrator(3)

for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, p, integrator)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_nodes()
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_error()
    rguh = ralg.simple_average(uh)
    eta = fem.recover_estimate(rguh)
    errorMatrix[3, i] = fem.get_recover_error(rguh)
    markedCell = mark(eta,theta=theta)
    if i < maxit - 1:
        markedCell = mark(eta,theta=theta)
        mesh.bisect(markedCell)

mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
cell = mesh.ds.cell
axes.plot_trisurf(x, y, cell, fem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

