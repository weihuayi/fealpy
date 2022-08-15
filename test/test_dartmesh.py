import numpy as np

from fealpy.mesh import DartMesh3d, TetrahedronMesh, MeshFactory

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float_)
cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
mesh = TetrahedronMesh(node, cell)

#mesh = MeshFactory.boxmesh3d([0, 1, 0, 1, 0, 1], nx=5, ny=5, nz=5, meshtype='hex')

mesh = DartMesh3d.from_mesh(mesh)
mesh = mesh.dual_mesh()
mesh.celldata['idx'] = np.arange(4)

mesh.to_vtk('test.vtu')


