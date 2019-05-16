import numpy as np
from fealpy.mesh.TriangleMesh import TriangleMesh


class EigenLShape2d:
    """
    example 4.1

    -\Delta u = \labmda u in \Omega 
    u = 0 on \partial \Omega

    \Omega = [-1, 1]*[-1, 1]\(0, 1]*(0, 1]
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
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


class EigenSquareDC:
    """
    example 4.2

    -\nabla\cdot (\nabla A\nabla u) = \labmda u in \Omega 
    u = 0 on \partial \Omega

    \Omega = [-1, 1]*[-1, 1]

    A = 100 x*y > 0
    A = 1  x*y < 0
    """
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


class EigenGWWA:
    """
    example 4.3

    -\Delta u = \labmda u in \Omega 
    u = 0 on \partial \Omega

    \Omega = [-1, 1]*[-1, 1]\(0, 1]*(0, 1]
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (1, -3),
            (-1, -1),
            (1, -1),
            (3, -1),
            (-3, 1),
            (-1, 1),
            (1, 1),
            (3, 1),
            (-1, 3)], dtype=np.float)

        cell = np.array([
            (2, 0, 3),
            (5, 4, 1),
            (5, 1, 6),
            (2, 6, 1),
            (6, 2, 7),
            (3, 7, 2),
            (5, 8, 4)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)

        return mesh

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)


class EigenGWWB:
    """
    example 4.4

    -\Delta u = \labmda u in \Omega 
    u = 0 on \partial \Omega

    \Omega = [-1, 1]*[-1, 1]\(0, 1]*(0, 1]
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        """isospectral domains A
        """
        node = np.array([
            (1, -3),
            (-1, -1),
            (1, -1),
            (3, -1),
            (-3, 1),
            (-1, 1),
            (1, 1),
            (-3, 3),
            (-1, 3)], dtype=np.float)

        cell = np.array([
            (2, 1, 0),
            (5, 4, 1),
            (5, 1, 6),
            (2, 6, 1),
            (2, 3, 6),
            (5, 8, 4),
            (7, 4, 0)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)

        return mesh

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)


