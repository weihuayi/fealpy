import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.vem_space import VirtualElementSpace2d
from fealpy.functionspace.function import FiniteElementFunction

from fealpy.mesh import rectangledomainmesh
from fealpy.mesh import topolygonmesh

from fealpy.form.vem import LaplaceSymetricForm
from fealpy.form.vem import SourceForm
from fealpy.boundarycondition import DirichletBC

from fealpy.solver import solve

from fealpy.model.poisson_model_2d import CosCosData, PolynomialData, ExpData

def isBoundaryDof(p):
    eps = 1e-14 
    return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)

p = 1 # degree of the vem space
box = [0, 1, 0, 1] # domain 
n = 10 # initial 
maxit = 5  
error = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

model = CosCosData()
model = PolynomialData()
model = ExpData()
for i in range(maxit):
    mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
    mesh = topolygonmesh(mesh)
    V = VirtualElementSpace2d(mesh, p) 
    a = LaplaceSymetricForm(V)
    L = SourceForm(V, model.source)

    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs() 

    BC = DirichletBC(V, model.dirichlet, isBoundaryDof)
    solve(a, L, uh, BC, 'direct')

    uI = V.interpolation(model.solution)
    error[i] = np.sqrt(np.sum((uh - uI)**2)/Ndof[i])
    n = 2*n


print(Ndof)
print(error)
print(error[:-1]/error[1:])
