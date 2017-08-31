import matplotlib.pyplot as plt
import numpy as np
import sys

from fealpy.mesh.meshio import load_mat_mesh
from fealpy.mesh.simple_mesh_generator import squaremesh
from fealpy.mesh.simple_mesh_generator import triangle, unitsquaredomainmesh

from fealpy.functionspace.tools import function_space 
from fealpy.femmodel.BiharmonicFEMModel import BiharmonicRecoveryFEM  
from fealpy.boundarycondition.BoundaryCondition import DirichletBC
from fealpy.solver import solve1
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error, H1_semi_error
from fealpy.model.BiharmonicModel2d import SinSinData, BiharmonicData2, BiharmonicData3, BiharmonicData4

m = int(sys.argv[1]) 
sigma = int(sys.argv[2])  

meshtype = int(sys.argv[3])

if m == 1:
    model = SinSinData()
elif m == 2:
    model = BihamonicData2(1.0,1.0)
elif m == 3:
    model = BihamonicData3()
elif m == 4:
    model = BihamonicData4()

maxit = 4
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
H1Serror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

if meshtype == 1:
    mesh = squaremesh(0, 1, 0, 1, r=6)
elif (meshtype == 3) or (meshtype == 4):
    box = [0, 1, 0, 1]
    h0 = 0.02
    mesh = triangle(box, h0)

for i in range(maxit):
    if meshtype == 2:
        mesh = load_mat_mesh('../data/square'+str(i+2)+'.mat')
    elif meshtype == 5:
        mesh = load_mat_mesh('../data/sqaureperturb'+str(i+2)+'.'+str(0.5) + '.mat')

    V = function_space(mesh, 'Lagrange', degree)
    V2 = function_space(mesh, 'Lagrange_2', degree)
    uh = FiniteElementFunction(V)
    ruh = FiniteElementFunction(V2)

    fem = BihamonicRecoveryFEM(V, model, sigma=sigma, rtype='simple')
    bc = DirichletBC(V, model.dirichlet, model.is_boundary_dof)
    solve1(fem, uh, dirichlet=bc, solver='cg')
    fem.recover_grad(uh, ruh)

    Ndof[i] = V.number_of_global_dofs() 
    error[i] = L2_error(model.solution, uh, order=4)
    derror[i] = div_error(model.laplace, ruh, order=4)
    gerror[i] = L2_error(model.gradient, ruh, order=5)
    H1Serror[i] = H1_semi_error(model.gradient, uh, order=5)

    if (meshtype == 1) or (meshtype == 3):
        mesh.uniform_refine()
    elif meshtype == 4:
        mesh = triangle(box, h0/(2**(i+1)))

print(Ndof)
print('L2 error:\n', error)
order = np.log(error[0:-1]/error[1:])/np.log(2)
print('order:\n', order)

print('div error:\n', derror)
order = np.log(derror[0:-1]/derror[1:])/np.log(2)
print('order:\n', order)

print('revover gradient error:\n', gerror)
order = np.log(gerror[0:-1]/gerror[1:])/np.log(2)
print('order:\n', order)

print('gradient error:\n', H1Serror)
order = np.log(H1Serror[0:-1]/H1Serror[1:])/np.log(2)
print('order:\n', order)
