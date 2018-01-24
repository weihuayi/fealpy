import numpy as np
import sys

from fealpy.model.poisson_model_2d import LShapeRSinData, CosCosData, KelloggData
from fealpy.vemmodel import PoissonVEMModel 
from fealpy.mesh.adaptive_tools import AdaptiveMarker 
from fealpy.tools.show import showmultirate
from fealpy.functionspace import FunctionNorm
from fealpy.quadrature import QuadrangleQuadrature 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = int(sys.argv[1])
maxit = int(sys.argv[2])
theta = float(sys.argv[3])

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
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
             '$\|\\nabla \Pi^\Delta u_h - \Pi^\Delta G(\\nabla \Pi^\Delta u_h) \|$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = quadtree.to_pmesh()

integrator = QuadrangleQuadrature(3)
vem = PoissonVEMModel(model, mesh, p=1)
funNorm = FunctionNorm(integrator, vem.area)
for i in range(maxit):
    print('step:', i)
    vem.solve()
    eta = vem.recover_estimate()
    Ndof[i] = vem.V.number_of_global_dofs()
    errorMatrix[0, i] = funNorm.l2_error(model.solution, vem.uh)
    e = vem.uh - vem.uI
    S = vem.project_to_smspace()
    errorMatrix[1, i] = np.sqrt(e@vem.A@e)
    errorMatrix[2, i] = funNorm.L2_error(model.solution, S.value, mesh=quadtree, barycenter=False)
    errorMatrix[3, i] = funNorm.L2_error(model.gradient, S.grad_value, mesh=quadtree, barycenter=False)
    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        quadtree.refine(marker=AdaptiveMarker(eta, theta=theta))
        vem.reinit(quadtree.to_pmesh())
        funNorm.area = vem.area

mesh = vem.V.mesh

fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
mesh.add_plot(axes, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh, cmap=plt.cm.jet, lw=0.0)


fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, k, Ndof, errorMatrix[:4, :], optionlist[:4], errorType[:4])
axes.legend(loc=3, prop={'size': 30})
plt.show()

