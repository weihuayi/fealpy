import numpy as np

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh

class LeftRightData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh


    @cartesian
    def pressure(self, p):
        pass

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def flux(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = np.abs(x) < 1e-13
        val[flag0] = 10
        flag1 = np.abs(x-1) < 1e-13
        val[flag1] = -10
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass

class CornerData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh


    @cartesian
    def pressure(self, p):
        pass

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def flux(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = 10

        flag0 = (np.abs(x-1) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x > 1 - 1/16)
        val[flag1] = -10
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass

class HoleData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=2, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(2)

        threshold = lambda p: (p[:, 0] > 0.25) & (p[:, 0] < 0.75) & (p[:, 1] >
                0.25) & (p[:, 1] < 0.75) 
        mesh.delete_cell(threshold)
        mesh.uniform_refine(n=n)
        return mesh


    @cartesian
    def pressure(self, p):
        pass

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def flux(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = 10

        flag0 = (np.abs(x-1) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x > 1 - 1/16)
        val[flag1] = -10
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass

class CrackData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=2, meshtype='tri'):
        node = np.array([
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (1.0, 1.0)], dtype=np.float)
        cell = np.array([
            (3, 0, 4), (1, 4, 0),
            (4, 1, 5), (2, 5, 1),
            (6, 3, 7), (4, 7, 3),
            (8, 4, 9), (5, 9, 4)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh 

    @cartesian
    def pressure(self, p):
        pass

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def flux(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = 10

        flag0 = (np.abs(x-1) < 1e-13) & (y > 1 - 1/16)
        flag1 = (np.abs(y-1) < 1e-13) & (x > 1 - 1/16)
        val[flag1] = -10
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x - 0.5) < 1e-13) & (y > 0.5)
        return flag
