import numpy as np

from ..mesh.TriangleMesh import TriangleMesh  
from ..mesh.tree_data_structure import Quadtree 

class PolyData:
    def __init__(self, beta=10):
        self.beta = beta


    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([
            (0, 0),
            (0.5, 0),
            (1, 0),
            (0, 0.5),
            (0.5, 0.5),
            (1, 0.5),
            (0, 1),
            (0.5, 1),
            (1, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([
                (0, 3, 4, 1),
                (1, 4, 5, 2),
                (3, 6, 7, 4),
                (4, 7, 8, 5)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
        elif meshtype is 'tri':
            cell = np.array([
                (1, 0, 4),
                (3, 4, 0),
                (4, 5, 1),
                (2, 1, 5), 
                (4, 3, 7),
                (6, 7, 3),
                (7, 8, 4),
                (5, 4, 8)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
        else:
            raise ValueError("".format)
        return mesh


        def source(self, p):
            """ The right hand side of DarcyForchheimer equation
            """
            x = p[..., 0]
            y = p[..., 1]
            pi = np.pi
            rhs = 2*pi*np.cos(pi*x)*np.sin(pi*y)
            return rhs


        def velocity_u(self, p):
            x = p[..., 0]
            y = p[..., 1]
            pi = np.pi
            val = np.sin(pi*x)*np.cos(pi*y)
            return val

        def velocity_v(self, p):
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
