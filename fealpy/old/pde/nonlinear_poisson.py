import numpy as np

from ..mesh import TriangleMesh, StructureQuadMesh

class CosCosData():
    """
    -\Delta u  + u**3 = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)+(np.cos(pi*x)*np.cos(pi*y))**3
        return val

    def dirichlet(self, p):
        return self.solution(p)

class SinSinData:
    """
    -\Delta u + u**2 = f
    u = sin(pi*x)*sin(pi*y)
    """
    def __init__(self, box):
        self.box = box

    def init_mesh(self, nx, ny):
        """
        generate the initial mesh
        """
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh

    def solution(self, p):
        """
        The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi

        val = np.sin(pi*x)*np.sin(pi*y)
        return val

    def source(self, p):
        """
        The right hand side of nonlinear poisson equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi

        rhs = 2*pi**2*np.sin(pi*x)*np.sin(pi*y) + \
                np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*y)

        return rhs

    def gradient(self, p):
        """
        The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi

        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        uprime[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return uprime

    def dirichlet(self, p):
        """
        Dirichlet boundary condition
        """
        return self.solution(p)
        
