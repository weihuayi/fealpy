import numpy as np
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
class CosCosCosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='Tet'):
        node = np.array([
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0, 0,-1)], dtype=np.float)

        cell = np.array([
            (0, 1, 2, 3),
            (0, 2, 1, 4)], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh
        
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return u
    
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

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val 

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


