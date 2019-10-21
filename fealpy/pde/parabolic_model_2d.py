import numpy as np

from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Quadtree import Quadtree
from ..mesh.QuadrangleMesh import QuadrangleMesh
from ..mesh.Tritree import Tritree
from ..mesh.StructureQuadMesh import StructureQuadMesh


class SinCosExpData:
    def __init__(self):
        self.diffusion_coefficient = 1.0/3.0

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def init_value(self, p):
        return self.solution(p, 0.0)

    def diffusion_coefficient(self):
        return self.diffusionCoefficient

    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        u = np.sin(pi*x)*np.cos(pi*y)*np.exp(-pi**2/8*t)
        return u

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        return 0.0

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return 0.0 

    def neuman(self, p, t):
        """ Neuman boundary condition
        """
        return 0.0

    def is_dirichlet_boundary(self, p):
        eps = 1e-14 
        return (p[:, 0] < eps) | (p[:, 0] > 1.0 - eps) 

    def is_neuman_boundary(self, p):
        eps = 1e-14
        return (p[:, 1] < eps) | (p[:, 1] > 1.0 - eps)

