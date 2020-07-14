import numpy as np
from scipy.spatial import KDTree

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

    def refine_boundary(self, n=1):
        mesh = self.mesh

        for i in range(n):
            halfedge = mesh.ds.halfedge

            NHE = len(halfedge) 
            isMarkedHEdge = np.ones(NHE, dtype=np.bool_)
            mesh.refine_halfedge(isMarkedHEdge)



    def run(self):
        self.refine_boundary(n=5)


