
from fealpy.np.mesh.triangle_mesh import TriangleMesh

nx = 2
ny = 2
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
NN = mesh.number_of_nodes()
print("NN:", NN)
NC = mesh.number_of_cells()
print("NC:", NC)
node = mesh.entity('node')
print("node:", node)
cell = mesh.entity('cell')
print("cell:", cell)
