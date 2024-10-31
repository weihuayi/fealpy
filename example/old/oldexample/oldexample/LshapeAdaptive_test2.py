import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_model_2d import LShapeRSinData, KelloggData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh.Tritree import Tritree

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

theta = 0.45
errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u- u_h ||_{0}$',
             '$|| \\nabla u - \\nabla u_h ||_{0}$',
             '$|| \\nabla u - G(\\nabla u_h) ||_{0}$',]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
integrator = mesh.integrator(3)
ralg = FEMFunctionRecoveryAlg()

tmesh = Tritree(mesh.node, mesh.ds.cell)
pmesh = tmesh.to_conformmesh()
for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, pmesh, p, integrator)
    fem.solve()
    uh = fem.uh
    rguh = ralg.simple_average(uh)
    eta = fem.recover_estimate(rguh)
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_error()
    errorMatrix[3, i] = fem.get_recover_error(rguh)
    #isMarkedCell = tmesh.refine_marker(eta, theta, method='L2')
    if i < maxit -1:
        isMarkedCell = tmesh.refine_marker(eta, theta, method='L2')
        tmesh.refine(isMarkedCell)
        pmesh = tmesh.to_conformmesh()
pmesh.add_plot(plt, cellcolor='w')
fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection = '3d')
x = pmesh.node[:, 0]
y = pmesh.node[:, 1]
cell = pmesh.ds.cell
axes.plot_trisurf(x, y, cell, fem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

























