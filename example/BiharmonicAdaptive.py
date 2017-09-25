
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
from fealpy.model.BiharmonicModel2d import BiharmonicData4, BiharmonicData5, BiharmonicData6, BiharmonicData7

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
    e = np.sqrt(e)
    #e = smooth_eta(mesh, e)
    return e 

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
        raise ValueError("I have note code the method")
    markedCell, = np.nonzero(isMarked)
    return markedCell


m = int(sys.argv[1])
theta = 0.3

if m == 1:
    model = BiharmonicData5(a=0.01)
    mesh = model.init_mesh()
elif m == 2:
    model = BiharmonicData6()
    mesh = model.init_mesh(n=4)
elif m == 3:
    model = BiharmonicData7()
    mesh = model.init_mesh(n=4)
else:
    raise ValueError("error!")

#model = BiharmonicData4()
#box = [0, 1, 0, 1]


sigma = 1
maxit = 40 
k = 20 
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
H1Serror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)



for i in range(maxit):
    print(i, 'step:')
    V = function_space(mesh, 'Lagrange', degree)
    V2 = function_space(mesh, 'Lagrange_2', degree)
    uh = FiniteElementFunction(V)
    ruh = FiniteElementFunction(V2)

    fem = BiharmonicRecoveryFEMModel(V, model, sigma=sigma, rtype='inv_area')
    bc = DirichletBC(V, model.dirichlet)
    solve(fem, uh, dirichlet=bc, solver='direct')
    fem.recover_grad(uh, ruh)

    Ndof[i] = V.number_of_global_dofs() 
    error[i]= L2_error(model.solution, uh, order=6)
    derror[i] = div_error(model.laplace, ruh, order=6)
    gerror[i] = L2_error(model.gradient, ruh, order=6)
    H1Serror[i] = H1_semi_error(model.gradient, uh, order=5)

    eta = estimate(uh, ruh)
    markedCell = mark(mesh, eta, theta)

    if i < maxit - 1:
        mesh.bisect(markedCell)


fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
#mesh.add_plot(axes, cellcolor=[1, 1, 1])
mesh.add_plot(axes)
fig1.savefig('mesh.pdf')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
axes.plot_trisurf(x, y, uh, triangles=mesh.ds.cell, cmap=plt.cm.jet)
fig2.savefig('solution.pdf')

fig3 = plt.figure()
fig3.set_facecolor('white')
axes = fig3.gca()
showrate(axes, k, Ndof, error, 'r-*', label='$||u - u_h||$')
showrate(axes, k, Ndof, H1Serror, 'b-o', label='$||\\nabla u - \\nabla u_h||$')
showrate(axes, k, Ndof, gerror, 'k-^', label='$||\\nabla u - G(\\nabla u_h)||$')
fig3.savefig('error.pdf')
plt.show()
