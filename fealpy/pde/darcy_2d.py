import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData:
    def __init__(self,box,nx,ny):
        self.box = box
        self.nx = nx
        self.ny = ny
        pass
    def init_mesh(self, n=2, meshtype ='quad'):
        nx = self.nx
        ny = self.ny
        box = self.box
        node = np.array([
            (0,0),
            (0, 0.5),
            (0,1),
            (0.5, 0),
            (0.5, 0.5),
            (0.5, 1),
            (1, 0),
            (1, 0.5),
            (1, 1)], dtype = np.float)
        cell = np.array([(0, 3, 1, 4),
            (1, 4, 5, 2),
            (3, 6, 7, 4),
            (4, 7, 5, 8)],dtype = np.int)
        mesh = StructureQuadMesh(box,nx,ny)
        return mesh
    def solution1(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*np.sin(2*pi*x)*np.cos(2*pi*y)
        return val

    def solution2(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = 2*pi*cos(2*pi*x)*sin(2*pi*y)
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

    def dirichlet(self, p):
        """ The dirichlet boundary condition
        """
        return self.pressure(p)

    def neumann(self, p):
        """ The neumann boundary condition
        """
        pass
