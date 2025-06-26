import numpy as np
from fealpy.mesh import QuadrangleMesh

class PoissonFvmPde1():
    def __init__(self):
        pass

    def mesh(self, nx, ny):
        self.nx = nx
        self.ny = ny
        mesh = QuadrangleMesh.from_box(box=[0,1,0,1], nx=nx, ny=ny)
        self.mesh = mesh
        return mesh

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return -np.exp(x + y)
    
    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return 2*np.exp(x + y)
    def dirichlet(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return -np.exp(x + y)
    
    def is_dirichlet_boundary(self, p):
        eps = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(y-1)<eps)|(np.abs(x-1)<eps)|(np.abs(x)<eps)|(np.abs(y)<eps)

