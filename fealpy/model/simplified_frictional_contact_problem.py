
import numpy as np

from ..mesh import TriangleMesh, QuadrangleMesh
from ..mesh.tree_data_structure import Quadtree

class SFContactProblemData:
    def __init__(self, eta=0.4):
        self.eta = eta 
    def init_mesh(self, n=4, meshtype='quadtree'):
        point = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quad':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = QuadrangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

    def solution(self, p):
        return np.zeros(p.shape[0:-1]) 

    def gradient(self, p):
        return np.zeros(p.shape) 

    def source(self, p):
        """The right hand side
        """
        x = p[..., 0]
        y = p[..., 1]
        val = 40*x - 20*y - 10 
        return val

    def dirichlet(self, p):
        return np.zeros(p.shape[0:-1]) 

    def is_dirichlet(self, p):
        eps = 1e-14 
        x = p[..., 0]
        y = p[..., 1]
        return (x < eps) | (y > 1.0 - eps) | (x > 1.0 - eps)

    def is_contact(self, p):
        eps = 1e-14 
        x = p[..., 0]
        y = p[..., 1]
        return y < eps 

class SFContactProblemData1:
    def __init__(self):
        self.eta = 1 

    def init_mesh(self, n=4, meshtype='quadtree'):
        point = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quad':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = QuadrangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

    def solution(self, p):
        return np.zeros(p.shape[0:-1]) 

    def gradient(self, p):
        return np.zeros(p.shape) 

    def source(self, p):
        """The right hand side
        """
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt((x - 0.8)**2 + (y + 0.2)**2)
        val =  (r**6*((20*r - 10.0)**2 + 1)**2*np.arctan(20*r - 10.0) - 40*r**5*((20*r - 10.0)**2 + 1) 
                + r**4*(16000*r - 8000.0)*(x - 0.8)**2 + r**4*(16000*r - 8000.0)*(y + 0.2)**2 
                + 20*r**3*(x - 0.8)**2*((20*r - 10.0)**2 + 1) + 20*r**3*(y + 0.2)**2*((20*r - 10.0)**2 + 1))/(r**6*((20*r - 10.0)**2 + 1)**2)
        return val

    def dirichlet(self, p):
        return np.zeros(p.shape[0:-1]) 

    def is_dirichlet(self, p):
        return np.zeros(p.shape[0:-1], dtype=np.bool) 

    def is_contact(self, p):
        eps = 1e-14 
        x = p[..., 0]
        y = p[..., 1]
        return (x < eps) | (y > 1.0 - eps) | (x > 1.0 - eps) | (y < eps)
