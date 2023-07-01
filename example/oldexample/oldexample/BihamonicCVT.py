import matplotlib.pyplot as plt
import numpy as np
import sys
from fealpy.mesh.meshio import load_mat_mesh
from fealpy.functionspace.tools import function_space 
from fealpy.form.Form import BihamonicRecoveryForm, SourceForm
from fealpy.boundarycondition.BoundaryCondition import DirichletBC, BihamonicRecoveryBC1
from fealpy.solver import solve
from fealpy.functionspace import Interpolation
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error
from fealpy.functionspace.tools import recover_grad
from fealpy.model.BihamonicModel2d import SinSinData, BihamonicData2, BihamonicData4

m = int(sys.argv[1]) 
sigma = int(sys.argv[2])  

if m == 1:
    model = SinSinData()
elif m == 2:
    model = BihamonicData2(1.0,1.0)
elif m == 3:
    model = BihamonicData4()

maxit = 4
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

for i in range(0, maxit):
    mesh = load_mat_mesh('../data/square'+str(i+2)+'.mat')
    V = function_space(mesh, 'Lagrange', degree)
    Ndof[i] = V.number_of_global_dofs() 
    uh = FiniteElementFunction(V)

    a  = BihamonicRecoveryForm(V, sigma=sigma)
    L = SourceForm(V, model.source, 4)

    bc0 = DirichletBC(V, model.dirichlet, model.is_boundary_dof)

    bc1 = BihamonicRecoveryBC1(V, model.neuman, sigma=sigma)

    solve(a, L, uh, dirichlet=bc0, neuman=bc1, solver='direct')
    error[i] = L2_error(model.solution, uh, order=4)
    ruh = recover_grad(uh)
    derror[i] = div_error(model.laplace, ruh, order=4)
    gerror[i] = L2_error(model.gradient, ruh, order=5)

#    if i < maxit-1:
#        mesh.uniform_refine()

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

