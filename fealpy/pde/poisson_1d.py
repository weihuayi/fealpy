import numpy as np
from fealpy.mesh.StructureIntervalMesh import StructureIntervalMesh
from fealpy.decorator import cartesian, barycentric

class CosData:
    def __init__(self):
        pass
    
    def domain(self):
        return np.array([0, 2])

    def init_mesh(self, n=1):
        mesh = StructureIntervalMesh(self.I, nx = self.nx)
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
        val = np.cos(np.pi*p)
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
        val = np.pi**2*np.cos(np.pi*p)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

