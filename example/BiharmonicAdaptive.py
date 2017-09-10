
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
from fealpy.model.BiharmonicModel2d import BiharmonicData4, BiharmonicData5, BiharmonicData6

from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 


def showrate(axes, k, N, error, option):
    axes.loglog(N, error, option, lw=2)
    c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)
    s = 0.75*error[0]/N[0]**c[0] 
    axes.loglog(N, s*N**c[0], label='N^'+str(c[0]))
    axes.legend()

def estimate(uh, ruh, order=2, dtype=np.float):
    V = uh.V
    mesh = V.mesh

    NC = mesh.number_of_cells()
    qf = TriangleQuadrature(order)
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    
    e = np.zeros((NC,), dtype=dtype)
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        uhval = uh.grad_value(lambda_k)
        ruhval = ruh.value(lambda_k)
        e += w_k*((uhval - ruhval)*(uhval - ruhval)).sum(axis=1)
    e *= mesh.area()
    return np.sqrt(e) 

def mark(mesh, eta, theta, method='L2'):
    NC = mesh.number_of_cells()
    isMarked = np.zeros(NC, dtype=np.bool)
    if method is 'L2':
        isMarked[eta > theta*np.max(eta)] = True
    else:
        raise ValueError("I have note code the method")
    markedCell, = np.nonzero(isMarked)
    return markedCell

theta = 0.6 

#model = BiharmonicData5()
#box = [-1, 1, -1, 1]

#model = BiharmonicData4()
#box = [0, 1, 0, 1]

model = BiharmonicData6()

sigma = 1
maxit = 100 
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
H1Serror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

#n = 40
#mesh = rectangledomainmesh(box, nx=n, ny=n)  

mesh = model.init_mesh(n=4)

for i in range(maxit):
    V = function_space(mesh, 'Lagrange', degree)
    V2 = function_space(mesh, 'Lagrange_2', degree)
    uh = FiniteElementFunction(V)
    ruh = FiniteElementFunction(V2)

    fem = BiharmonicRecoveryFEMModel(V, model, sigma=sigma, rtype='inv_area')
    bc = DirichletBC(V, model.dirichlet)
    solve(fem, uh, dirichlet=bc, solver='direct')
    fem.recover_grad(uh, ruh)

    Ndof[i] = V.number_of_global_dofs() 
    error[i], _ = L2_error(model.solution, uh, order=6)
    derror[i] = div_error(model.laplace, ruh, order=6)
    gerror[i], _ = L2_error(model.gradient, ruh, order=6)
    H1Serror[i], _ = H1_semi_error(model.gradient, uh, order=5)

    eta = estimate(uh, ruh)
    markedCell = mark(mesh, eta, theta)

    if i < maxit - 1:
        mesh.bisect(markedCell)


fig = plt.figure()
axes = fig.add_subplot(1, 3, 1)
mesh.add_plot(axes)

axes = fig.add_subplot(1, 3, 2, projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
axes.plot_trisurf(x, y, uh, triangles=mesh.ds.cell, cmap=plt.cm.Spectral)

axes = fig.add_subplot(1, 3, 3)
showrate(axes, 10, Ndof, error, 'r-*')
showrate(axes, 10, Ndof, H1Serror, 'b-o')
plt.show()
