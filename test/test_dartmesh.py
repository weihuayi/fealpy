import numpy as np

from fealpy.mesh import DartMesh3d, TetrahedronMesh, MeshFactory

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
#node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
#        dtype=np.float_)
#cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
mesh = TetrahedronMesh(node, cell)

mesh = MeshFactory.boxmesh3d([0, 1, 0, 1, 0, 1], nx=2, ny=1, nz=1, meshtype='tet')

mesh = DartMesh3d.from_mesh(mesh)
mesh.celldata['idx'] = np.arange(mesh.number_of_cells())
mesh.to_vtk('init.vtu')

mesh = mesh.dual_mesh(dual_point='barycenter')
mesh.celldata['idx'] = np.arange(mesh.number_of_cells())
#print(np.where(mesh.ds.dart[:, 2]==0))
mesh.to_vtk('dual.vtu')


