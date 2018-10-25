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
        rhs = np.zeros(p.shape[0],)

        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = -y
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = 2/np.pi*np.arctan(10*(x+y-1))
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2+y**2)
        val = -m*y + 20/(np.pi*(1+(10*(x+y-1))**2))
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2 + y**2)
        val = m*x + 20/(np.pi*(1+(10*(x+y-1))**2))
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 20/(np.pi*(1+(10*(x+y-1))**2))
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 20/(np.pi*(1+(10*(x+y-1))**2))
        return val
