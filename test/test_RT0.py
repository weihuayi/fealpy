import numpy as np

from fealpy.functionspace import RTFiniteElementSpace2d
from fealpy.mesh import TriangleMesh


node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float)
cell = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int)


mesh = TriangleMesh(node, cell)

space = RTFiniteElementSpace2d(mesh, p=0)

bc = np.array([[1/3, 1/3, 1/3]], dtype=np.float)
bc = np.array([[0.0, 0.5, 0.5]], dtype=np.float)


phi = space.basis( bc )
print( phi )
