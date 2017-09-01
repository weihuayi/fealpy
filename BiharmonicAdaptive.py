
import matplotlib.pyplot as plt
import numpy as np
import sys

from fealpy.mesh.meshio import load_mat_mesh
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh  
from fealpy.mesh.simple_mesh_generator import triangle, unitsquaredomainmesh

from fealpy.functionspace.tools import function_space 
from fealpy.femmodel.BiharmonicFEMModel import BiharmonicRecoveryFEM  
from fealpy.boundarycondition.BoundaryCondition import DirichletBC
from fealpy.solver import solve1
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis.PrioriError import L2_error, div_error, H1_semi_error
from fealpy.model.BiharmonicModel2d import BiharmonicData5 


def mark(mesh, eta, theta, method='L2'):
    NC = mesh.number_of_cells()
    isMarked = np.zeros(NC, dtype=np.bool)
    if method is 'L2':
        isMarked[eta > theta*np.max(eta)] = True
    else:
        raise ValueError("I have note code the method")

model = BiharmonicData5()

maxit = 4
degree = 1
error = np.zeros((maxit,), dtype=np.float)
derror = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
H1Serror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)


for i in range(maxit):
    V = function_space(mesh, 'Lagrange', degree)
    V2 = function_space(mesh, 'Lagrange_2', degree)
    uh = FiniteElementFunction(V)
    ruh = FiniteElementFunction(V2)
    fem = BiharmonicRecoveryFEM(V, model, sigma=sigma, rtype='inv_area')
    bc = DirichletBC(V, model.dirichlet, model.is_boundary_dof)
    solve1(fem, uh, dirichlet=bc, solver='direct')
    fem.recover_grad(uh, ruh)

    Ndof[i] = V.number_of_global_dofs() 
    error[i] = L2_error(model.solution, uh, order=4)
    derror[i] = div_error(model.laplace, ruh, order=4)
    gerror[i] = L2_error(model.gradient, ruh, order=5)
    H1Serror[i] = H1_semi_error(model.gradient, uh, order=5)

