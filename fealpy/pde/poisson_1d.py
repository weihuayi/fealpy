import numpy as np
from fealpy.mesh.IntervalMesh import IntervalMesh

class CosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1):
        node = np.array([[0], [1]], dtype=np.float)
        cell = np.array([(0, 1)], dtype=np.int)
        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        """ the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        x = p[..., 0]
        val = np.cos(np.pi*x)
        return u

    def gradient(self, p):
        """ The gradient of the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        val = -np.pi*np.sin(pi*p)
        return val

    def source(self, p):
        x = p[..., 0]
        val = np.pi**2*np.cos(np.pi*x)
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

