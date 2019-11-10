import numpy as np
from fealpy.mesh.IntervalMesh import IntervalMesh

class CosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1):
        node = np.array([0, 1], dtype=np.float)
        cell = np.array([(0, 1)], dtype=np.int)
        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        """ the exact solution
        """
        u = np.cos(np.pi*p)
        return u

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = -pi*sin(pi*p)
        return val

    def source(self, p):
        val = np.pi**2*np.cos(np.pi*p)
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

