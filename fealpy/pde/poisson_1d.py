import numpy as np
from fealpy.mesh import IntervalMesh
from fealpy.decorator import cartesian, barycentric

class CosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1):
        node = np.array([[0], [1]], dtype=np.float64)
        cell = np.array([(0, 1)], dtype=np.int_)
        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh
    @cartesian
    def solution(self, p):
        """ the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        x = p[..., 0]
        val = np.cos(np.pi*x)
        return val 

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        pi = np.pi
        val = -pi*np.sin(pi*p)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        val = np.pi**2*np.cos(np.pi*x)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

