
import numpy as np
import sys 
import matplotlib.pyplot as plt

from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.functionspace.tools import function_space
from fealpy.form.Form import LaplaceSymetricForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.function import FiniteElementFunction
from fealpy.erroranalysis import L2_error
from fealpy.model.poisson_model_2d import CosCosData

from scipy.sparse.linalg import inv, dsolve, spsolve
from scipy.sparse import spdiags, bmat, tril, triu, diags

def sinsin(n, point):
    x = point[:, 0]
    y = point[:, 1]
    pi = np.pi
    return np.sin(n*pi*x)*np.sin(n*pi*y)

def prolongate_matrix(mesh, n):
    point = mesh.point
    N = mesh.number_of_points()
    I = spdiags(np.ones(N, dtype=np.float), 0, N, N)
    P = np.zeros((N, n-1), dtype=np.float)

    for i in range(1, n):
        P[:, i-1] = sinsin(i, point)

    return bmat([[I, P]])


box = [0, 1, 0, 1]
n = 6 
model = CosCosData()
mesh = rectangledomainmesh(box, nx=2**n, ny=2**n, meshtype='tri') 
N = mesh.number_of_points()
V = function_space(mesh, 'Lagrange', 1)
point = V.interpolation_points()

uh = FiniteElementFunction(V)
a  = LaplaceSymetricForm(V, 3)
L = SourceForm(V, model.source, 3)
bc = DirichletBC(V, model.dirichlet, model.is_boundary)

A = a.get_matrix()
b = L.get_vector()
A, b = bc.apply(A, b)

PI = prolongate_matrix(mesh, n)
AA = PI.transpose()@A@PI
bb = PI.transpose()@b

DL = tril(AA).tocsc()
DLInv = inv(DL)
U = triu(AA, 1)

x0 = np.zeros(N+n-1, dtype=np.float)

for i in range(20):
    x1 = DLInv@(-U@x0 + bb)
    e = np.linalg.norm(x1 - x0)/np.linalg.norm(x0)
    print(e)
    x0 = x1
    
