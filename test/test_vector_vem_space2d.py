import numpy as np

import matplotlib.pyplot as plt
from fealpy.functionspace.vector_vem_space import VectorScaledMonomialSpace2d
from fealpy.functionspace.vector_vem_space import VectorVirtualElementSpace2d
from fealpy.mesh import PolygonMesh
from fealpy.vem.integral_alg import PolygonMeshIntegralAlg


def u(point):
    x = point[..., 0]
    y = point[..., 1]
    val = np.zeros(point.shape, dtype=np.float)
    val[..., 0] = np.exp(x)
    val[..., 1] = np.exp(y)
    return val

node = np.array([ (0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float)
cell = np.array([0, 1, 2, 3], dtype=np.int)
cellLocation = np.array([0, 4], dtype=np.int)

mesh = PolygonMesh(node, cell, cellLocation)

p = 2
q = 3
vem = VectorVirtualElementSpace2d(mesh, p)

integralalg = PolygonMeshIntegralAlg(
   mesh.integrator(q), 
   mesh, 
   area=vem.vsmspace.area, 
   barycenter=vem.vsmspace.barycenter)

uI = vem.interpolation(u, integralalg.integral)
print(uI)
print(u(node))

