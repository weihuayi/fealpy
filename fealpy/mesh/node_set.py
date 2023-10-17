import numpy as np
from scipy.spatial import cKDTree
from .uniform_mesh_2d import UniformMesh2d
from .uniform_mesh_3d import UniformMesh3d

class NodeSet:
    def __init__(self, node, capacity=None):
        self.node = node
        self.tree = cKDTree(node)

        self.nodedata = {}

    def dimension(self):
        return self.node.shape[1]

    def number_of_nodes(self):
        return len(self.node)

    def add_node_data(self, name, dtype=np.float64):
        NN = self.number_of_nodes()
        self.nodedata[name] = np.zeros(NN, dtype=dtype)

    def node_data(self, name, subname=None):
        if subname is None:
            return self.nodedata[name]
        else:
            return self.nodedata[name][subname]

    def add_plot(self, axes, color='k', markersize=20):
        axes.set_aspect('equal')
        return axes.scatter(self.node[..., 0], self.node[..., 1], c=color, s=markersize)

    def neighbors(self, h, points=None):
        if points is None:
            return self.tree.query_ball_point(self.node, h)
        else:
            return self.tree.query_ball_point(points, h)


    @classmethod
    def from_dam_break_domain(cls, dx=0.02, dy=0.02):
        nx = int(1/dx) + 1
        ny = int(2/dy) + 1
        mesh = UniformMesh2d([1, nx, 1, ny], h=(dx, dy), origin=(dx, dy))

        wnode = mesh.entity('node')

        return cls(wnode)


