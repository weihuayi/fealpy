import numpy as np
from scipy.spatial import KDTree, Delaunay

class TriAdvancingFrontAlg():

    def __init__(self, mesh):
        """

        Parameters
        ----------
        mesh ：HalfEdgeMesh2d object
        Notes
        -----

        输入一个初始的半边网格，利用波前法插入三角形网格单元，生成最终的三角形网
        格。
        """
        self.mesh = mesh
        self.front = None
        self.w = np.array([[0, 1], [-1, 0]])  # 逆时针 90 度旋转矩阵

    def uniform_refine_boundary(self, n=1):
        mesh = self.mesh
        for i in range(n):
            halfedge = mesh.ds.halfedge
            NHE = len(halfedge) 
            isMarkedHEdge = np.ones(NHE, dtype=np.bool_)
            mesh.refine_halfedge(isMarkedHEdge)

    def advance(self):
        mesh = self.mesh
        halfedge = mesh.ds.halfedge
        subdomain = mesh.ds.subdomain
        node = mesh.node


        isFront = subdomain[halfedge[:, 1]] > 0
        idx0 = halfedge[halfedge[isFront, 3], 0] # 前一条半边的顶点编号
        idx1 = halfedge[isFront, 0] # 当前半边的顶点编号
        v = node[idx1] - node[idx0]
        n = v@self.w
        fnode = (node[idx1] + node[idx0])/2
        fnode += np.sqrt(3)/2*n

        NN = len(fnode)
        he2fnode = np.zeros(len(halfedge), dtype=np.int_)
        he2fnode[isFront] = range(NN)

        isKeepFNode = np.zeros(NN, dtype=np.bool_)

        idx = he2fnode[halfedge[isFront, 2]]
        h = np.sqrt(np.sum((fnode - fnode[idx])**2, axis=-1))

        return fnode

    def run(self):
        self.uniform_refine_boundary(n=5)

        node = self.mesh.entity('node')

        tri = Delaunay(node, incremental=True)

        fnode = self.advance()

        tri.add_points(fnode)

        return tri, fnode






        


