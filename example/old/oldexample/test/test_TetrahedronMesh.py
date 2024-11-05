from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
import numpy as np

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)

mesh = TetrahedronMesh(node, cell)
mesh.uniform_refine(5)
p = 4

space = LagrangeFiniteElementSpace(mesh, p=p)

c2p = space.dof.cell_to_dof()
c2p1 = mesh.cell_to_ipoint_1(p)
print(c2p)
print(c2p1)
print(np.sum(np.abs(c2p-c2p1)))



