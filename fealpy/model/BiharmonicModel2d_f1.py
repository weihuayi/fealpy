import numpy as np
from ..mesh.TriangleMesh import TriangleMesh  

class KDomain:
    def __init__(self):
        pass

    def init_mesh(self, n=2):
        point = np.array([
            (0, 0), (1, 0), (2, 0), (3, 0), (5, 0),
            (0, 1), (1, 1), (2, 1), (4, 1), 
            (0, 2), (1, 2), (2, 2), (3, 2),
            (0, 3), (1, 3), (2, 3), (4, 3),
            (0, 4), (1, 4), (2, 4), (3, 4), (5, 4)], dtype=np.float)
        cell = np.array([
            (1, 6, 0), (5, 0, 6), (2, 7, 1), (6, 1, 7), (3, 8, 7), (8, 3, 4),
            (6, 10, 5), (9, 5, 10), (7, 11, 6), (10, 6, 11), (11, 7, 12), (12, 7, 8),
            (10, 14, 9), (13, 9, 14), (11, 15, 10), (14, 10, 15), (11, 12, 15), (12, 16, 15), 
            (14, 18, 13), (17, 13, 18), (15, 19, 14), (18, 14, 19), (20, 15, 16), (16, 21, 20)], dtype=np.int)
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
        val = np.zeros(p.shape, dtype=np.dtype)
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
