import numpy as np
from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Quadtree import Quadtree
from ..mesh.Tritree import Tritree


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
        mesh.uniform_bisect(2)
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
            (7, 4, 8)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_bisect(2)
        mesh.uniform_refine(n)

        return mesh

    def solution(self, p):
        return 0

    def dirichlet(self, p):
        return self.solution(p)


class EigenCrack:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri'):
        if meshtype == 'tri':
            node = np.array([
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (0, 1)], dtype=np.float)

            cell = np.array([
                (2, 1, 0),
                (2, 0, 3),
                (2, 5, 1),
                (2, 4, 5)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            r = 1-2**(1/2)/2
            a = 1/2 - 2**(1/2)/2
            rr = 1/2
            node = np.array([
                (0, -1),
                (-rr, -rr),
                (rr, -rr),
                (-r, -r),
                (0, -r),
                (r, -r),
                (-1, 0),
                (-r, 0),
                (0, 0),
                (r, 0),
                (1, 0),
                (r, 0),
                (-r, r),
                (0, r),
                (r, r),
                (-rr, rr),
                (rr, rr),
                (0, 1)], dtype=np.float)
            cell = np.array([
                (0, 4, 3, 1),
                (2, 5, 4, 0),
                (1, 3, 7, 6),
                (3, 4, 8, 7),
                (4, 5, 9, 8),
                (5, 2, 10, 9),
                (6, 7, 12, 15),
                (7, 8, 13, 12),
                (8, 11, 14, 13),
                (11, 10, 16, 14),
                (12, 13, 17, 15),
                (13, 14, 16, 17)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            node = np.array([
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (0, 1)], dtype=np.float)

            cell = np.array([
                (2, 1, 0),
                (2, 0, 3),
                (2, 5, 1),
                (2, 4, 5)], dtype=np.int)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def domain(self, n):
        pass 

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        u = np.sqrt(1/2*(r - x)) - 1/4*r**2
        return u

    def source(self, p):
        rhs = np.ones(p.shape[0:-1])
        return rhs

    def gradient(self, p):
        """the gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -0.5*x + (-0.5*x + 0.5*r)**(-0.5)*(0.25*x/r - 0.25)
        val[..., 1] = 0.25*y*(-0.5*x + 0.5*r)**(-0.5)/r - 0.5*y

        return val

    def dirichlet(self, p):
        return self.solution(p)

