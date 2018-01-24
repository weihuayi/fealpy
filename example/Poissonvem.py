import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import rectangledomainmesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityPoint
from fealpy.mesh.PolygonMesh import PolygonMesh

from fealpy.vemmodel import PoissonVEMModel

from fealpy.model.poisson_model_2d import CosCosData, PolynomialData, ExpData


p = 1 # degree of the vem space
box = [0, 1, 0, 1] # domain 
n = 10 # initial 
maxit = 5  
error = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

model = CosCosData()
#model = PolynomialData()
#model = ExpData()
for i in range(maxit):
    # Mesh 
    mesh0 = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
    mesh1 = TriangleMeshWithInfinityPoint(mesh0)
    point, cell, cellLocation = mesh1.to_polygonmesh()
    mesh = PolygonMesh(point, cell, cellLocation)

    vem = PoissonVEMModel(model, mesh, p=1)
    vem.solve()
    # VEM Space
    print(vem.V.smspace.area[:10])

    Ndof[i] = len(vem.uh) 

    # error 
    error[i] = vem.l2_error() 
    n = 2*n


print(Ndof)
print(error)
print(error[:-1]/error[1:])
