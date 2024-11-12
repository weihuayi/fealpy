from  meshpy.triangle import MeshInfo, build
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
domain = MeshInfo()
domain.set_points([(0,0),(1,0),(1,1),(0,1)])
domain.set_facets([(0,1),(1,2),(2,3),(3,0)], facet_markers=[1, 2, 3, 4])
mesh = build(domain, max_volume = 0.1**2, attributes=True)
node = np.array(mesh.points, dtype = np.float)
cell = np.array(mesh.elements, dtype = np.int)
tmesh = TriangleMesh(node, cell)

fig = plt.figure()
axes =  fig.gca()
tmesh.add_plot(axes)

cell = tmesh.entity('cell')
node = tmesh.entity('node')
NN = tmesh.number_of_nodes()
isBdNode = tmesh.ds.boundary_node_flag()

newNode = np.zeros((NN, 2), dtype=np.float)
degree = np.zeros(NN, dtype=np.int)
np.add.at(degree, cell, 1)
for i in range(10):
    #bc = tmesh.entity_barycenter('cell')
    bc, R = tmesh.circumcenter()
    np.add.at(newNode, (cell, np.s_[:]), bc[:, np.newaxis, :])
    newNode /= degree[:, np.newaxis]
    node[~isBdNode] = newNode[~isBdNode]
    newNode[:] = 0

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, node=newNode, color='r')
tmesh.find_node(axes, node=bc, color='b')
plt.show()

