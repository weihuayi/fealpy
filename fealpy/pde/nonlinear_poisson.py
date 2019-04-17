import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

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
        

