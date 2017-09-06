import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.vem_space import VirtualElementSpace2d
from fealpy.functionspace.function import FiniteElementFunction

from fealpy.mesh import rectangledomainmesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityPoint
from fealpy.mesh.PolygonMesh import PolygonMesh

from fealpy.vemmodel import PoissonVEMModel
from fealpy.boundarycondition import DirichletBC

from fealpy.solver import solve

from fealpy.model.poisson_model_2d import CosCosData, PolynomialData, ExpData


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
    # Mesh 
    mesh0 = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
    mesh1 = TriangleMeshWithInfinityPoint(mesh0)
    point, cell, cellLocation = mesh1.to_polygonmesh()
    mesh = PolygonMesh(point, cell, cellLocation)

    # VEM Space
    V = VirtualElementSpace2d(mesh, p) 

    uh = FiniteElementFunction(V)

    Ndof[i] = V.number_of_global_dofs() 

    vem = PoissonVEMModel(model, V)
    BC = DirichletBC(V, model.dirichlet)

    solve(vem, uh, dirichlet=BC, solver='direct')

    # error 
    uI = V.interpolation(model.solution)
    error[i] = np.sqrt(np.sum((uh - uI)**2)/Ndof[i])
    n = 2*n


print(Ndof)
print(error)
print(error[:-1]/error[1:])
