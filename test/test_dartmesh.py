import numpy as np

from fealpy.mesh import DartMesh3d, TetrahedronMesh

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
mesh = TetrahedronMesh(node, cell)

mesh = DartMesh3d.from_mesh(mesh)

