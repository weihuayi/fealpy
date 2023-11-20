import numpy as np
from scipy.spatial import cKDTree
from .uniform_mesh_2d import UniformMesh2d
from .uniform_mesh_3d import UniformMesh3d

class NodeSet:
    def __init__(self, node, capacity=None):
        self.node = node
        self.tree = cKDTree(node)
        self.is_boundary = np.zeros(self.number_of_nodes(), dtype=bool)
        self.nodedata = {}

    def dimension(self):
        return self.node.shape[1]

    def number_of_nodes(self):
        return len(self.node)

    def is_boundary_node(self):
        return self.is_boundary

    def add_node_data(self, name ,dtype=np.float64):
        NN = self.number_of_nodes()
        self.nodedata.update({names:np.zeros(NN,dtypes) 
            for names,dtypes in zip(name,dtype)})
        
    def node_data(self, name):
        return self.nodedata[name]
    
    def set_node_data(self, name, val):
        self.nodedata[name][:] = val

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
        pp = np.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T
        
        #下
        bp0 = np.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
        bp1 = np.mgrid[-dx/2:4+dx/2:dx, -dy/2:dy/2:dy].reshape(2, -1).T
        bp = np.vstack((bp0,bp1))
         
        #左
        lp0 = np.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
        lp1 = np.mgrid[-dx/2:dx/2:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        lp = np.vstack((lp0,lp1))
        
        #右
        rp0 = np.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
        rp1 = np.mgrid[4+dx/2:4+dx:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        rp = np.vstack((rp0,rp1))
        
        boundaryp = np.vstack((bp,lp,rp))
        node = np.vstack((pp,boundaryp))
 
        result = cls(node)
        result.is_boundary[pp.shape[0]:] = True 
        return result


