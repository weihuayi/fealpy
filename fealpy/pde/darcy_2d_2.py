import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val[..., 0] = -y*x*(1-x)
        val[..., 1] = x*y*(1-y)
        return val

    def velocity_x(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        val = -y*x*(1-x)
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        val = x*y*(1-y)
        return val

    def pressure(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.exp(x*(1-x)*y*(1-y))
        return val

    def source1(self, p):
        """ The right hand of Darcy equation
        """
        x = p[..., 0]
        y = p[..., 1]
        val = x - y
        return val

    def source2(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = -self.mu/self.k*x*y*(1-x) + (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def source3(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = self.mu/self.k*x*y*(1-y) + x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def g_D(self,p):
        val = np.zeros((p.shape[0],1))
        return val


