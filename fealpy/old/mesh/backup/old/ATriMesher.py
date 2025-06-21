import numpy as np
from scipy.spatial import KDTree


class ATriMesher:

    def __init__(self, domain):
        """
        Parameters
        ----------
        domain : HalEdgeDomain
        """
        self.domain = domain


        self.maxNN = 100000  
        self.maxNE = 100000
        self.maxNC = 100000
        self.GD = 2
        self.node = np.zeros((self.maxNN, self.GD), dtype=np.float)
        self.halfedge = np.zeros((2*self.maxNE, ), dtype=np.int)

        self.node = domain.vertices.copy()
        self.halfedge = domain.halfedge.copy()
        self.subdomain, _, j = np.unique(halfedge[:, 1],
            return_index=True, return_inverse=True)
        self.halfedge[:, 1] = j

    def uniform_boundary_meshing(self, refine=4, maxh=0.1):
        self.domain.boundary_uniform_refine(n=refine)

    def advance(self):

        w = np.array([[0, 1], [-1, 0]])  # 逆时针 90 度旋转矩阵

        node = self.domain.vertices
        halfedge = self.domain.halfedge
       
        isIHEdge = halfedge[:, 1] > 0 # 区域内部的半边
        idx0 = halfedge[halfedge[isIHEdge, 3], 0] # 前一条半边的顶点编号
        idx1 = halfedge[isIHEdge, 0] # 当前半边的顶点编号
        v = node[idx1] - node[idx0]
        n = v@w
        anode = (node[idx1] + node[idx0])/2
        anode += np.sqrt(3)/2*n

        # 修正角点相邻点的半径， 如果角点的角度小于 theta 的
        # 这里假设角点相邻的节点， 到角点的距离相等
        isFixed = self.domain.fixed[halfedge[isIHEdge, 0]]
        idx, = np.nonzero(isFixed)
        pre = halfedge[idx, 3]
        nex = halfedge[idx, 2]

        p0 = node[halfedge[pre, 0]]
        p1 = node[halfedge[idx, 0]]
        p2 = node[halfedge[nex, 0]]

        v0 = p2 - p1
        v1 = p0 - p1
        l0 = np.sqrt(np.sum(v0**2, axis=-1))
        l1 = np.sqrt(np.sum(v1**2, axis=-1))
        s = np.cross(v0, v1)/l0/l1
        c = np.sum(v0*v1, axis=-1)/l0/l1
        a = np.arcsin(s)
        a[s < 0] += 2*np.pi
        a[c == -1] = np.pi
        a = np.degrees(a)

        return anode
