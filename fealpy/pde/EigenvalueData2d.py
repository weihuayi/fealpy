import numpy as np
from fealpy.mesh.TriangleMesh import TriangleMesh


class EigenLShape2d:
    def __init__(self):
        pass

    def diffusion_coefficient(self, p):
        return 1

    def init_mesh(self, n=1, meshtype='tri'):
        """ L-shaped domain
        \Omega = [-1, 1]*[-1, 1]\(0, 1]*(0, 1]
        """
        node = np.array([
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([
            (1, 4, 0),
            (3, 0, 4),
            (2, 5, 1),
            (4, 1, 5),
            (4, 7, 3),
            (6, 3, 7)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)

class EignSquareDC:
    def __init__(self, a=100):
        self.a = a

    def init_mesh(self, n=1, meshtype='tri'):
        """ Quad shape domain
        \Omega = [-1, 1]*[-1, 1]
        """
        node = np.array([
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)

        cell = np.array([
            (1, 4, 0),
            (3, 0, 4),
            (2, 5, 1),
            (4, 1, 5),
            (4, 7, 3),
            (6, 3, 7),
            (5, 8, 4),
            (7, 4, 8)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def diffusion_coefficient(self, p):
        idx = (p[..., 0]*p[..., 1] > 0)
        k = np.ones(p.shape[:-1], dtype=np.float)
        k[idx] = self.a
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        is_subdomain = [p[..., 0]*p[..., 1] > 0, p[..., 0]*p[..., 1] < 0]
        return is_subdomain

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)

class example3_A:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        """isospectral domains A
        """
        node = np.array([
            (1, -3),
            (1, -2),
            (2, -2),
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (3, -1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (-3, 1),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (-2, 2),
            (-1, 2),
            (-1, 3)], dtype=np.float)

        cell = np.array([
            (1, 0, 2),
            (1, 2, 5),
            (6, 5, 2),
            (6, 2, 7),
            (9, 8, 3),
            (4, 10, 3),
            (9, 3, 10),
            (4, 5, 10),
            (11, 10, 5),
            (6, 12, 5),
            (11, 5, 12),
            (6, 7, 12),
            (13, 12, 7),
            (15, 14, 8),
            (9, 16, 8),
            (15, 8, 16),
            (9, 10, 16),
            (17, 16, 10),
            (11, 18, 10),
            (17, 10, 18),
            (11, 12, 18),
            (19, 18, 12),
            (13, 20, 12),
            (19, 12, 20),
            (15, 21, 14),
            (15, 16, 21),
            (22, 21, 16),
            (22, 23, 21)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)

        return mesh

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)


class example3_B:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (1, -3),
            (0, -2),
            (1, -2),
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (3, -1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (-3, 1),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (-3, 2),
            (-2, 2),
            (-1, 2),
            (-3, 3),
            (-2, 3),
            (-1, 3)], dtype=np.float)

        cell = np.array([
            (2, 1, 0),
            (4, 3, 1),
            (2, 5, 1),
            (4, 1, 5),
            (9, 8, 3),
            (4, 10, 3),
            (9, 3, 10),
            (4, 5, 10),
            (11, 10, 5),
            (6, 12, 5),
            (11, 5, 12),
            (6, 7, 12),
            (14, 13, 8),
            (9, 15, 8),
            (14, 8, 15),
            (9, 10, 15),
            (16, 15, 10),
            (11, 17, 10),
            (16, 10, 17),
            (11, 12, 17),
            (14, 19, 13),
            (18, 13, 19),
            (14, 15, 19),
            (20, 19, 15),
            (18, 19, 21),
            (22, 21, 19),
            (20, 23, 19),
            (22, 19, 23)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)

        return mesh
    
    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)
