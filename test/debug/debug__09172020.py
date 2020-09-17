import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import PolygonMesh,HalfEdgeMesh2d

node3 = np.array([
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
    (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)
], dtype = np.float64)
cell3 = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4 ,4, 7, 8, 5], dtype = np.int)
cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype = np.int)   

mesh3 = PolygonMesh(node3, cell3, cellLocation)
mesh3 = HalfEdgeMesh2d.from_mesh(mesh3)
mesh3.uniform_refine(2)                                            # 这里一致加密的时候会报错
node3 = mesh3.entity("node")
edge3 = mesh3.entity("edge")
cell3, cellLocation = mesh3.entity("cell")
halfedge = mesh3.entity("halfedge")

fig3 = plt.figure()
axes3 = fig3.gca()
mesh3.add_plot(axes3)
mesh3.find_node(axes3, showindex=True)
mesh3.find_edge(axes3)
plt.show()