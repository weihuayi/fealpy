import numpy as np
from ..mesh.TriangleMesh import TriangleMesh  

class KDomain:
    def __init__(self):
        pass

    def init_mesh(self, n=2):
        point = np.array([
            (0, 0), (1, 0), (2, 0), (3, 0), 
            (0, 1), (1, 1), (2, 1),  
            (0, 2), (1, 2), 
            (0, 3), (1, 3), (2, 3), 
            (0, 4), (1, 4), (2, 4), (3, 4)], dtype=np.float)
        cell = np.array([
            (1, 5, 0), (4, 0, 5), (6, 5, 2), (2, 3, 6), 
            (5, 8, 4), (7, 4, 8), (5, 6, 8),
            (8, 10, 7), (9, 7, 10), (10, 8, 11),
            (12, 9, 13), (10, 13, 9), (11, 14, 10), (14, 11, 15)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n=n)
        return mesh 


    def domain(self):
        a = np.sqrt(2)
        point = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 0),
            (a, 0),
            (1.5, 2),
            (a, 4),
            (2, 4),
            (1, 3),
            (1, 4),
            (0, 4)], dtype=np.float)
        segment = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 0)], dtype=np.float)
        return point, segment

    def gradient(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        val = np.zeros(p.shape[0], dtype=np.float)
        return val

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = np.zeros(p.shape[0], dtype=np.float)
        return val

    def source(self, p):
        return 1
