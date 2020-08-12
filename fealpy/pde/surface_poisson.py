import numpy as np

from fealpy.decorator import cartesian

class SphereSinSinSinData():
    def __init__(self):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()

    def domain(self):
        return self.surface

    def init_mesh(self, n=0):
        mesh = self.surface.init_mesh()
        mesh.uniform_refine(n, self.surface)
        return mesh

    @cartesian
    def solution(self,p):
        """ The exact solution
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z)
        return u

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        t1 = sin(pi*x)*sin(pi*y)*sin(pi*z)*pi
        t2 = sin(pi*x)*cos(pi*z)*cos(pi*y)*pi*y*z + sin(pi*y)*cos(pi*z)*cos(pi*x)*pi*x*z + cos(pi*x)*sin(pi*z)*cos(pi*y)*pi*x*y
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)*z + sin(pi*x)*sin(pi*z)*cos(pi*y)*y + sin(pi*y)*cos(pi*x)*sin(pi*z)*x
        r = x**2 + y**2 + z**2
        rhs = 2*pi*(t1 + (t2 + t3)/r) 
        return rhs

    @cartesian
    def gradient(self, p):
        """ The Gradu of the exact solution
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        r = x**2 + y**2 + z**2
        
        t1 = cos(pi*x)*sin(pi*y)*sin(pi*z)
        t2 = sin(pi*x)*cos(pi*y)*sin(pi*z)
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)

        valx = pi*(t1 - (t1*x**2 + t2*x*y + t3*x*z)/r)
        valy = pi*(t2 - (t1*x*y + t2*y**2 + t3*y*z)/r)
        valz = pi*(t3 - (t1*x*z + t2*y*z +t3*z**2)/r)
        
        grad = np.zeros(p.shape, dtype=np.float64)
        grad[..., 0] = valx
        grad[..., 1] = valy
        grad[..., 2] = valz
        return grad  
