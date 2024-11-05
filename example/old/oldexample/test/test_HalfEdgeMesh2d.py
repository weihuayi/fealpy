import numpy as np
from fealpy.mesh import HalfEdgeMesh2d, TriangleMesh
import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build
import copy

def f(node):
    node = node/np.linalg.norm(node, axis=-1).reshape((node.shape[:-1]+(1, )))
    return node

n = 20
theta = np.linspace(0, np.pi*2, n, endpoint=False)
node = np.c_[np.cos(theta), np.sin(theta)]
line = np.c_[np.arange(n), (np.arange(n)+1)%n]
mesh_int = MeshInfo()
mesh_int.set_points(node)
mesh_int.set_facets(line)


mesh = build(mesh_int, max_volume = np.sqrt(3)*np.pi/640)
node = np.array(mesh.points, dtype =np.float)
cell = np.array(mesh.elements, dtype = np.int)

#node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
#cell = np.array([[0, 1, 2], [0, 2, 3]])

#node = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float_)
#cell = np.array([[0, 1, 2]])

mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
#mesh.tri_uniform_refine(4)
mesh0 = copy.deepcopy(mesh)
mesh.to_dual_mesh(projection=f)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#mesh.add_halfedge_plot(axes, showindex=True)

fig = plt.figure()
axes = fig.gca()
mesh0.add_plot(axes)
#mesh.add_halfedge_plot(axes, showindex=True)
plt.show()
