
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

from fealpy.mesh.meshio import load_mat_mesh
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh  
from fealpy.mesh.simple_mesh_generator import triangle, unitsquaredomainmesh

from fealpy.functionspace.tools import function_space 
from fealpy.femmodel.BiharmonicFEMModel import BiharmonicRecoveryFEMModel
from fealpy.boundarycondition.BoundaryCondition import DirichletBC
from fealpy.solver import solve
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error, H1_semi_error
from fealpy.model.BiharmonicModel2d_f1 import KDomain

from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

from fealpy.tools.show import showrate


def smooth_eta(mesh, eta):
    m = np.max(eta)
    eta /= m
    area = mesh.area()
    q2c = mesh.ds.point_to_cell()
    cell = mesh.ds.cell
    w = q2c@area
    for i in range(10): 
        beta = q2c@(eta*area)/w
        eta = np.sum(beta[cell], axis=1)/3 
    return eta
        
def mark(mesh, eta, theta, method='L2'):
    NC = mesh.number_of_cells()
    isMarked = np.zeros(NC, dtype=np.bool)
    if method is 'MAX':
        isMarked[eta > theta*np.max(eta)] = True
    elif method is 'L2':
        eta = eta**2
        idx = np.argsort(eta)[-1::-1]
        x = np.cumsum(eta[idx])
        isMarked[idx[x < theta*x[-1]]] = True
        isMarked[idx[0]] = True
    else:
        raise ValueError("I have not code the method")
    markedCell, = np.nonzero(isMarked)
    return markedCell


m = int(sys.argv[1])
theta = 0.2

if m == 1:
    model = KDomain()
    mesh = model.init_mesh(n=4)

sigma = 1
maxit = 40 
k = 20 
degree = 1

errorType = ['$\|\\nabla u_h - G(\\nabla u_h) \|$']

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)


for i in range(maxit):
    print(i, 'step:')
    V = function_space(mesh, 'Lagrange', degree)
    V2 = function_space(mesh, 'Lagrange_2', degree)
    uh = FiniteElementFunction(V)
    rgh = FiniteElementFunction(V2)
    rlh = FiniteElementFunction(V)

    fem = BiharmonicRecoveryFEMModel(V, model, sigma=sigma, rtype='inv_area')
    bc = DirichletBC(V, model.dirichlet)
    solve(fem, uh, dirichlet=bc, solver='direct')
    fem.recover_grad(uh, rgh)
    fem.recover_laplace(rgh, rlh)

    eta = fem.recover_estimate(uh, rgh)

    Ndof[i] = V.number_of_global_dofs() 
    errorMatrix[0, i] = np.sqrt(np.sum(eta**2))

    markedCell = mark(mesh, eta, theta)

    if i < maxit - 1:
        mesh.bisect(markedCell)


fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
#mesh.add_plot(axes, cellcolor=[1, 1, 1])
mesh.add_plot(axes, cellcolor='w')
fig1.savefig('mesh.pdf')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
axes.plot_trisurf(x, y, uh, triangles=mesh.ds.cell, cmap=plt.cm.jet, lw=0.)
fig2.savefig('solution.pdf')

fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
showrate(axes, k, Ndof, errorMatrix[0], 'k-*', label=errorType[0])
fig3.savefig('error.pdf')
plt.show()
