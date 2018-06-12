import numpy as np
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
from scipy.optimize import minimize

class TriRadiusRatioQuality():
    def __init__(self, mesh):
        self.mesh = mesh;

    def __call__(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)
        q = self.quality(node)
        return np.mean(q) 

    def get_init_value(self):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        x0 = node[~isBdNode].reshape(-1)
        return x0


    def callback(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node.copy()
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)
        flag = self.is_valid(node)
        q = self.quality(node)
        print('Max quality:', q.max())
        print('All area > 0:', flag)
        return False 

    def update_mesh_node(self, x):
        NN = self.mesh.number_of_nodes()
        node = self.mesh.node
        isBdNode = self.mesh.ds.boundary_node_flag()
        NB = isBdNode.sum()
        node[~isBdNode] = x.reshape(NN - NB, 2)

    def quality(self, node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l = np.zeros((NC, 3))
        for i in range(3):
            l[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2 
        quality = p*q/(16*area**2)
        return quality

    def is_valid(self, node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        area = np.cross(v[2], -v[1])/2 
        return np.all(area > 0)


h = 0.05
mesh_info = MeshInfo()

# Set the vertices of the domain [0, 1]^2
mesh_info.set_points([
    (0,0), (1,0), (1,1), (0,1)])

# Set the facets of the domain [0, 1]^2
mesh_info.set_facets([
    [0,1],
    [1,2],
    [2,3],
    [3,0]
    ])

# Generate the tet mesh
mesh = build(mesh_info, max_volume=(h)**2)
node = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)
tmesh = TriangleMesh(node, cell)
quality = TriRadiusRatioQuality(tmesh)

x0 = quality.get_init_value()
R = minimize(quality, x0, method='L-BFGS-B', callback=quality.callback)
print(R)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')

quality.update_mesh_node(R.x)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')

plt.show()
