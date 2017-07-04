import matplotlib.pyplot as plt
import numpy as np

from fealpy.Mesh import rectangledomainmesh
from fealpy import function_space
from fealpy import LaplaceSymetricForm, SourceForm
from fealpy import DirichletBC
from fealpy import solve
from fealpy import FiniteElementFunction
from fealpy import L2_error
from fealpy.Model import CosCosData
from scipy.sparse.linalg import eigsh

import sys


degree = int(sys.argv[1])

"""

"""


def isBoundaryDof(p):
    eps = 1e-14 
    return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)


model = CosCosData()
box = [0, 1, 0, 1]
qorder = 1
n = 8 
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
maxit = 8 
error = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    V = function_space(mesh, 'Lagrange', degree)
    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs() 
    a  = LaplaceSymetricForm(V, qorder)
    L = SourceForm(V, model.source, qorder)
    BC = DirichletBC(V, model.dirichlet, isBoundaryDof)
    A, b = BC.apply(a.get_matrix(), L.get_vector())
    w, v = eigsh(A, k=2, which='SM')
    print(w)
    if i < maxit-1:
        mesh.uniform_refine()

