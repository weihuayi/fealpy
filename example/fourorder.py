import matplotlib.pyplot as plt
import numpy as np

from fealpy.mesh.simple_mesh_generator import squaremesh
from fealpy.functionspace.tools import function_space 
from fealpy.form.Form import BihamonicRecoveryForm, SourceForm
from fealpy.boundarycondition.BoundaryCondition import DirichletBC
from fealpy.solver import solve
from fealpy.functionspace import Interpolation
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error
from fealpy.functionspace.tools import recover_grad
from fealpy.model.FourthOrderModel2d import PolynomialData, SinSinData


def isBoundaryDof(p):
    eps = 1e-14 
    return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)

model = SinSinData()
mesh = squaremesh(0, 1, 0, 1, r=3)
maxit = 4
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

for i in range(maxit):

    V = function_space(mesh, 'Lagrange', degree)
    Ndof[i] = V.number_of_global_dofs() 
    uh = FiniteElementFunction(V)

    a  = BihamonicRecoveryForm(V, sigma=2)
    L = SourceForm(V, model.source, 4)

    BC = DirichletBC(V, model.dirichlet, isBoundaryDof)
    solve(a, L, uh, BC, 'direct')
    error[i] = L2_error(model.solution, uh, order=4)
    ruh = recover_grad(uh)
    derror[i] = div_error(model.laplace, ruh, order=4)
    gerror[i] = L2_error(model.gradient, ruh, order=4)

    if i < maxit-1:
        mesh.uniform_refine()

print(Ndof)
print('L2 error:\n', error)
order = np.log(error[0:-1]/error[1:])/np.log(2)
print('order:\n', order)

print('div error:\n', derror)
order = np.log(derror[0:-1]/derror[1:])/np.log(2)
print('order:\n', order)

print('gradient error:\n', gerror)
order = np.log(gerror[0:-1]/gerror[1:])/np.log(2)
print('order:\n', order)
