import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.vem_space import VirtualElementSpace2d
from fealpy.functionspace.vem_space import ScaledMonomialSpace2d
from fealpy.functionspace.function import FiniteElementFunction

from fealpy.mesh import rectangledomainmesh
from fealpy.mesh.PolygonMesh import PolygonMesh

from fealpy.erroranalysis.PrioriError import L2_error

from fealpy.form.vem import LaplaceSymetricForm
from fealpy.form.vem import SourceForm
from fealpy.boundarycondition import DirichletBC

from fealpy.solver import solve

from fealpy.model.poisson_model_2d import CosCosData, PolynomialData, ExpData

from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityPoint


def tri_to_polygon(mesh):
    nmesh = TriangleMeshWithInfinityPoint(mesh)
    ppoint, pcell, pcellLocation =  nmesh.to_polygonmesh()
    return PolygonMesh(ppoint, pcell, pcellLocation)


p = 1 # degree of the vem space
box = [0, 1, 0, 1] # domain 
n = 10 # initial 
N  = (n+1)*(n+1)

maxit = 3

errorl2 = np.zeros((maxit,), dtype=np.float)
errorL2 = np.zeros((maxit,), dtype=np.float)
errorH1 = np.zeros((maxit,), dtype=np.float)
errorMax = np.zeros((maxit,), dtype=np.float)
gerror = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)
#model = CosCosData()
#model = PolynomialData()
model = ExpData()


# f = ['Square0.1.mat', 'Square0.05.mat',....]

for i in range(maxit):
    mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='polygon') 

    # tmesh = load_mat_mesh(f[i])
    # mesh = tri_to_polygon(tmesh)
    
    V = VirtualElementSpace2d(mesh, p) 
    M = ScaledMonomialSpace2d(mesh, p)
    h= M.h
    a = LaplaceSymetricForm(V)
    L = SourceForm(V, model.source)

    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs()
    point=V.interpolation_points()
   
    bc = DirichletBC(V, model.dirichlet, model.is_boundary)
    A = solve(a, L, uh, dirichlet=bc, solver='direct')
    uI = V.interpolation(model.solution)
    errorl2[i] = np.sqrt(np.sum((uh - uI)**2)/Ndof[i])
    errorH1[i] = np.sqrt((uh - uI)@A@(uh - uI))
    errorL2[i] =h[0]* np.sqrt(np.sum((uh - uI)**2))
    errorMax[i] = np.max(np.abs(uh-uI))
    n = 2*n

print(Ndof)

print('l2 error:\n', errorl2)
order = np.log(errorl2[:-1]/errorl2[1:])/np.log(2)
print('order:\n', order)

print('gradient error:\n', errorH1)
order = np.log(errorH1[0:-1]/errorH1[1:])/np.log(2)
print('order:\n', order)

print('L2 error:\n', errorL2)
order = np.log(errorL2[:-1]/errorL2[1:])/np.log(2)
print('order:\n', order)
  

print('Max error:\n', errorL2)
order = np.log(errorMax[:-1]/errorMax[1:])/np.log(2)
print('order:\n', order)

