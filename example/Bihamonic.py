import matplotlib.pyplot as plt
import numpy as np

from fealpy.mesh.simple_mesh_generator import squaremesh
from fealpy.functionspace.tools import function_space 
from fealpy.form.Form import BihamonicRecoveryForm, SourceForm
from fealpy.boundarycondition.BoundaryCondition import DirichletBC, BihamonicRecoveryBC1
from fealpy.solver import solve
from fealpy.functionspace import Interpolation
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error
from fealpy.functionspace.tools import recover_grad
from fealpy.model.BihamonicModel2d import SinSinData, BihamonicData2, BihamonicData4

model = SinSinData()
#model = BihamonicData2(1.0,1.0)
#model = BihamonicData4()
mesh = squaremesh(0, 1, 0, 1, r=6)
maxit = 4
degree = 1
sigma = 100
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

for i in range(maxit):

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

#fig, axes = plt.subplots(1, 2)
#
##fig = plt.figure()
##axes = fig.gca()
#
#ipoints = V.interpolation_points()
#cell2dof = V.cell_to_dof()
#u = model.solution(ipoints)
#ue = np.zeros(Ndof[-1], dtype=np.float)
#ue[:] = uh[:]
#
#e = np.sum(np.abs(u - ue)[cell2dof], axis=1)/3
#mesh.add_plot(axes[0], cellcolor=np.sum(u[cell2dof], axis=1)/3, showcolorbar=True)
#mesh.add_plot(axes[1], cellcolor=np.sum(ue[cell2dof], axis=1)/3, showcolorbar=True)
##mesh.add_plot(axes, cellcolor=e, showcolorbar=True)
#plt.show()
#
