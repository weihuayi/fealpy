import numpy as np
import sys

from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh 
from fealpy.mesh.PolygonMesh import PolygonMesh

from fealpy.functionspace.vem_space import VirtualElementSpace2d
from fealpy.functionspace.function import FiniteElementFunction

from fealpy.model.poisson_model_2d import LShapeRSinData, CosCosData, KelloggData
from fealpy.vemmodel import PoissonVEMModel 
from fealpy.boundarycondition import DirichletBC

from fealpy.solver import solve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showrate

def vem_solve(model, quadtree):
    mesh = quadtree.to_pmesh() 
    V = VirtualElementSpace2d(mesh, 1)
    uh = FiniteElementFunction(V)
    vem = PoissonVEMModel(model, V)
    BC = DirichletBC(V, model.dirichlet)
    A, b = solve(vem, uh, dirichlet=BC, solver='direct')
    uI = V.interpolation(model.solution)
    eta = vem.recover_estimate(uh)
    uIuh = np.sqrt((uh - uI)@A@(uh - uI))
    return uh, V.number_of_global_dofs(), eta, uIuh

class AdaptiveMarker():
    def __init__(self, eta, theta=0.2):
        self.eta = eta
        self.theta = theta

    def refine_marker(self, qtmesh):
        idx = qtmesh.leaf_cell_index()
        markedIdx = mark(self.eta, self.theta)
        return idx[markedIdx]

    def coarsen_marker(self, qtmesh):
        pass

m = int(sys.argv[1])

if m == 1:
    model = KelloggData()
    quadtree = model.init_mesh(n=3)
elif m == 2:
    model = LShapeRSinData() 
    quadtree = model.init_mesh(n=4)
elif m == 3:
    model = CosCosData()
    quadtree = model.init_mesh(n=4)

maxit = 30 
k = maxit - 10 
errorType = ['$\| u - u_h\|$',
             '$\|\\nabla u_I - \\nabla u_h\|$',
             '$\|\\nabla u - \\nabla u_h\|$',
             '$\|\\nabla u_h - G(\\nabla u_h) \|$',
             '$\|\\nabla u - G(\\nabla u_h)\|$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    uh, Ndof[i], eta, uIuh= vem_solve(model, quadtree)
    errorMatrix[1, i] = uIuh 
    errorMatrix[3, i] = np.sqrt(np.sum(eta*eta))
    if i < maxit - 1:
        quadtree.refine(marker=AdaptiveMarker(eta, theta=0.3))

mesh = uh.V.mesh
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
axes.plot_trisurf(x, y, uh, cmap=plt.cm.jet, lw=0.0)
fig2.savefig('solution.pdf')


fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
#showrate(axes, k, Ndof, errorMatrix[0], 'k-*', label=errorType[0])
showrate(axes, k, Ndof, errorMatrix[1], 'b-o', label=errorType[1])
#showrate(axes, k, Ndof, errorMatrix[2], 'r-^', label=errorType[2])
showrate(axes, k, Ndof, errorMatrix[3], 'g->', label=errorType[3]) 
#showrate(axes, k, Ndof, errorMatrix[4], 'm-8', label=errorType[4])
#showrate(axes, k, Ndof, errorMatrix[5], 'c-D', label=errorType[5])
plt.show()

#print(Ndof)
#print(error)
#print(error[:-1]/error[1:])
#print(rerror)
#print(rerror[:-1]/rerror[1:])
