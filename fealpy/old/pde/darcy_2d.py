import numpy as np

from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Mesh2d import Mesh2d

class CoscosData:
    def __init__(self, box,mu=1,k=1):
        self.box = box
        self.mu = 1
        self.k = 1

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh
        
#    def init_mesh(self, n=1, meshtype='tri'):
#        node = np.array([
#            (-1, -1),
#            (1, -1),
#            (1, 1),
#            (-1, 1)], dtype=np.float)

#        if meshtype is 'tri':
#            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
#            mesh = TriangleMesh(node, cell)
#            mesh.uniform_refine(n)
#            return mesh
#        else:
#            raise ValueError("".format)
        

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
        val = self.mu/self.k*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
        return val

    def source1(self, p):
        """ The right hand of Darcy equation
        """
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(np.pi)**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
        return val

    def source2(self,p):
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:, 0] = np.zeros(p.shape[0])
        val[:, 1] = np.zeros(p.shape[0])
        return val

    def source3(self,p):
        val = np.zeros(p.shape[0])
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

    def g_D(self,p):
        val = np.zeros((p.shape[0],1))
        return val

class PolynormialData:
    def __init__(self, box, mu, k):
        self.box = box
        self.mu = mu
        self.k = k

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val[..., 0] = np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = np.cos(pi*x)*np.sin(pi*y)
        return val

    def velocity_x(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val = np.sin(pi*x)*np.cos(pi*y)
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        val = np.cos(pi*x)*np.sin(pi*y)
        return val

    def pressure(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*y*(1-y)
        return val

    def source1(self, p):
        """ The right hand of Darcy equation
        """
        x = p[..., 0]
        y = p[..., 1]
        val = 2*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)
        return val

    def source2(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = self.mu/self.k*np.sin(np.pi*x)*np.cos(np.pi*y) \
                + (1-2*x)*y*(1-y)
        return val

    def source3(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = self.mu/self.k*np.cos(np.pi*x)*np.sin(np.pi*y) \
                + x*(1-x)*(1-2*y)
        return val

    def g_D(self,p):
        val = np.zeros((p.shape[0],1))
        return val

class ExponentData:
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
        val = np.exp(x*(1-x)*y*(1-y)) -1
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



