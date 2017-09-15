import numpy as np

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

def lshape_mesh(r=1):
    point = np.array([
        (-1, -1),
        (0, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)], dtype=np.float)
    cell = np.array([
        (0, 1, 3, 2),
        (2, 3, 6, 5),
        (3, 4, 7, 6)], dtype=np.int)
    return point, cell

def showrate(axes, k, N, error, option):
    axes.loglog(N, error, option, lw=2)
    c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)
    s = 0.75*error[0]/N[0]**c[0] 
    axes.loglog(N, s*N**c[0], label='N^'+str(c[0]))
    axes.legend()

def vem_solve(model, quadtree):
    mesh = quadtree.to_polygonmesh() 
    V = VirtualElementSpace2d(mesh, 1)
    uh = FiniteElementFunction(V)
    vem = PoissonVEMModel(model, V)
    BC = DirichletBC(V, model.dirichlet)
    solve(vem, uh, dirichlet=BC, solver='direct')
    eta = vem.recover_estimate(uh)
    return uh, V.number_of_global_dofs(), eta

class AdaptiveMarker():
    def __init__(self, eta, theta=0.5):
        self.eta = eta
        self.theta = theta

    def refine_marker(self, qtmesh):
        idx = qtmesh.leaf_cell_index()
        #m = np.max(eta)
        m = np.mean(eta)
        theta = self.theta
        return idx[eta > 1.8*m]


    def coarsen_marker(self, qtmesh):
        pass

point, cell = lshape_mesh(4)
quadtree = Quadtree(point, cell)
quadtree.uniform_refine(4)
#model = LShapeRSinData() 
#model = CosCosData()

maxit = 50
error = np.zeros((maxit,), dtype=np.float)
rerror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

for i in range(maxit):
    print('step:', i)
    uh, Ndof[i], eta= vem_solve(model, quadtree)
    uI = uh.V.interpolation(model.solution)
    error[i] = np.sqrt(np.sum((uh - uI)**2)/Ndof[i])
    rerror[i] = np.sqrt(np.sum(eta*eta))
    if i < maxit - 1:
        quadtree.refine(marker=AdaptiveMarker(eta, theta=0.8))


mesh = uh.V.mesh
fig = plt.figure()
axes = fig.add_subplot(1, 3, 1)
mesh.add_plot(axes, cellcolor=eta, showcolorbar=True)

axes = fig.add_subplot(1, 3, 2)
c2p = mesh.ds.cell_to_point()
#c = c2p@uh/c2p.sum(axis=1)
mesh.add_plot(axes)

axes = fig.add_subplot(1, 3, 3)
showrate(axes, 30, Ndof, error, 'r-*')
showrate(axes, 30, Ndof, rerror, 'b-o')
plt.show()

#print(Ndof)
#print(error)
#print(error[:-1]/error[1:])
#print(rerror)
#print(rerror[:-1]/rerror[1:])
