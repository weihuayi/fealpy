import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData1:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1
        self.rho = 1
        self.beta = 0
        self.tol = 1e-6


    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = x-y
        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = -y*x*(1-x)
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*y*(1-y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.exp(x*(1-x)*y*(1-y)) - 1
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = -(mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2)))
        val = m*x*y*(1-x) + (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2))
        val = m*x*y*(1-y) + x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y))
        return val
    def grad_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val[..., 0] = 1
        val[..., 1] = -1
        return val

    def dirichlet(self, p):
        """ Dirichlet boundary condition
        """
        val = np.zeros(p.shape[0],1)
        return val
