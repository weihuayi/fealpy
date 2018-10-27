import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData1:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1
        self.rho = 1
        self.beta = 5


    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p.shape[0]
        rhs = np.zeros(x,)
        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(x)*np.cos(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = -np.cos(x)*np.sin(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t0*t3 + pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = -m*t1*t2 + pi*np.sin(pi*x)*np.cos(pi*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val= pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = pi*np.sin(pi*x)*np.cos(pi*y)
        return val

