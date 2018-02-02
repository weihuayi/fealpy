import numpy as np
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.mesh import TriangleMesh 


point = np.array([
    [0,0],
    [1,0],
    [1,1],
    [0,1]], dtype=np.float)
cell = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int)

mesh = TriangleMesh(point, cell)
V = LagrangeFiniteElementSpace(mesh, p=3)
f = V.function(dim=3)
c = np.random.rand(f.shape[0], f.shape[1])
f += c
print(f.V)
