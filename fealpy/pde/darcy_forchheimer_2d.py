import numpy as np

from ..mesh.litest_quadmesh import StructureQuadMesh1
from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.Mesh2d import Mesh2d

class PolyData:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1
        self.rho = 1
        self.beta = 5
        self.tol = 1e-6
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
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
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = (1-x)*y*(1-y) - x*y*(1-y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-y) - x*(1-x)*y
        return val

    def norm_u(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        val = np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        return val

class ExponentData:
    def __init__(self, box):
        self.box = box
        self.mu = 2
        self.k = 1
        self.rho = 1
        self.beta = 5
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
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y))
        return val
    def norm_u(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2))
        return val

class SinsinData:
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

    def norm_u(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        val = np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        return val
class ArctanData:
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

    def norm_u(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sqrt(x**2 + y**2)
        return val
