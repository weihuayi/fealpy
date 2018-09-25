import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData:
    def __init__(self, box):
        self.box = box
        self.mu = 1
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
        val[..., 0] = 2*pi*np.sin(2*pi*x)*np.cos(2*pi*y)
        val[..., 1] = 2*pi*np.cos(2*pi*x)*np.sin(2*pi*y)
        return val

    def velocity_x(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val = 2*pi*np.sin(2*pi*x)*np.cos(2*pi*y)
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val = 2*pi*np.cos(2*pi*x)*np.sin(2*pi*y)
        return val

    def pressure(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
        return val

    def source(self, p):
        """ The right hand of Darcy equation
        """
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(np.pi)**2*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
        return val

    def gradient(self, p):
        """ The gradient of the exact pressure
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -2*pi*np.sin(2*pi*x)*np.cos(2*pi*y)
        val[..., 1] = -2*pi*np.cos(2*pi*x)*np.sin(2*pi*y)
        return val


