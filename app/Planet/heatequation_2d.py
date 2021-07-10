import numpy as np
from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh 


class CosCosExpData:
    def __init__(self, h):
        self.h = h

    def init_mesh(self, n=0):
        mesh = unitcircledomainmesh(self.h)          
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = (1/np.pi)*np.cos(pi*x)*np.cos(pi*y)*np.exp(-t)
        return u

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        val = np.array([0.0],dtype=p.dtype)
        dim = len(p.shape)-1
        val.reshape((1,)*dim)
        return val

    def neuman(self, p, t):
        """ Neuman boundary condition
        """
        return -1.0


