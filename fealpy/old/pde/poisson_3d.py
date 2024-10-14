import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.geometry.implicit_surface import SphereSurface

class SinSinSinData:
    def __init__(self):
        pass
    def domain(self):#我加的
        return np.array([0, 1, 0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = pi*cos(pi*x)*sin(pi*y)*sin(pi*z)
        val[..., 1] = pi*sin(pi*x)*cos(pi*y)*sin(pi*z)
        val[..., 2] = pi*sin(pi*x)*sin(pi*y)*cos(pi*z)
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
    
    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 3)
        n: (NE, 3)

        grad*n : (NQ, NE, 3)
        """
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

class CosCosCosData:
    def __init__(self):
        pass
    def domain(self):#我加的
        return np.array([0, 1, 0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -pi*sin(pi*x)*cos(pi*y)*cos(pi*z)
        val[..., 1] = -pi*cos(pi*x)*sin(pi*y)*cos(pi*z)
        val[..., 2] = -pi*cos(pi*x)*cos(pi*y)*sin(pi*z)
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
    
    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 3)
        n: (NE, 3)

        grad*n : (NQ, NE, 3)
        """
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa


class X2Y2Z2Data:
    def __init__(self):
        pass


    def domain(self):
        return [-1, 1, -1, 1, -1, 1]

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = x**2*y**2*z**2
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 2*x*y**2*z**2
        val[..., 1] = 2*x**2*y*z**2
        val[..., 2] = 2*x**2*y**2*z
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = -(2*y**2*z**2 + 2*x**2*z**2 + 2*x**2*y**2)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
    
    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 3)
        n: (NE, 3)

        grad*n : (NQ, NE, 3)
        """
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=2, meshtype='tet'):
        node = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]], dtype=np.float64)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int_)
        mesh = TetrahedronMesh(node, cell)
        for i in range(n):
            mesh.bisect()

        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        isDelCell = ((bc[:, 0] > 0) & (bc[:, 1] < 0))
        cell = cell[~isDelCell]
        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        node = node[isValidNode]

        idxMap = np.zeros(NN, dtype=mesh.itype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]
        mesh = TetrahedronMesh(node, cell)
        mesh.label() # 标记最长边

        return mesh

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2*pi
        u = r**(2/3)*np.sin(2*theta/3)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos

        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)

        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = (
                2/3*r**(-1/3)*sin(2*theta/3)*x/r -
                2/3*r**(2/3)*cos(2*theta/3)*y/r**2
                )
        val[..., 1] = (
                2/3*r**(-1/3)*sin(2*theta/3)*y/r +
                2/3*r**(2/3)*cos(2*theta/3)*x/r**2
                )
        return val

    @cartesian
    def source(self, p):
        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
