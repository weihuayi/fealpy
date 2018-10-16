import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData1:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1
        self.rho = 1
        self.beta = 20


    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = 2*pi*np.cos(pi*x)*np.cos(pi*y)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = np.sin(pi*x)*np.cos(pi*y)
        val[:,1] = np.cos(pi*x)*np.sin(pi*y)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.cos(pi*y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.sin(pi*y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*y*(1-y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t0*t3 + (1-2*x)*y*(1-y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t1*t2 +  x*(1-x)*(1-2*y)
        return val
    def grad_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val[..., 0] = (1-x)*y*(1-y) - x*y*(1-y)
        val[..., 1] = x*(1-x)*(1-y) - x*(1-x)*y
        return val

    def dirichlet(self, p):
        """ Dirichlet boundary condition
        """
        val = np.zeros(p.shape[0],1)
        return val
