from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm

mesh = TriangleMesh.from_box()
node = mesh.entity_barycenter('node')
c2n = mesh.cell_to_node()
print(node[c2n])

a = bm.array([1,2,3,4,5,6])
print(a.reshape(2, -1).T)

