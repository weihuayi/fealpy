import numpy as np
from ..timeintegratoralg.timeline import UniformTimeLine
from ..timeintegratoralg.timeline import ChebyshevTimeLine


class SinSinSinExpDataSphere():
    def __init__(self):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()

    def domain(self):
        return self.surface

    def time_mesh(self, t0, t1, NT, timeline='uniform'):
        if timeline is 'uniform':
            return UniformTimeLine(t0, t1, NT)
        elif timeline is 'chebyshev':
            return ChebyshevTimeLine(t0, t1, NT)

    def init_mesh(self, n=4):
        mesh = self.surface.init_mesh()
        mesh.uniform_refine(n, surface=self.surface)
        return mesh
    
    def solution(self, p, t):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z)*np.exp(-t)
        return u

    def source(self, p, t):
        """ The right hand side of parabolic equation
        INPUT:
            p: array object, N*3
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        r = x**2 + y**2 + z**2
        t1 = (-1 + 2*pi**2)*sin(pi*x)*sin(pi*y)*sin(pi*z)*np.exp(-t)
        t2 = 2*pi*np.exp(-t)*(
                sin(pi*x)*sin(pi*y)*cos(pi*z)*z +
                sin(pi*x)*sin(pi*z)*cos(pi*y)*y +
                sin(pi*y)*cos(pi*x)*sin(pi*z)*x)/r
        t3 = 2*pi**2*np.exp(-t)*(
                sin(pi*x)*cos(pi*z)*cos(pi*y)*y*z +
                sin(pi*y)*cos(pi*z)*cos(pi*x)*x*z +
                cos(pi*x)*sin(pi*z)*cos(pi*y)*x*y)/r

        rhs = t1 + t2 + t3
        return rhs
 
class ExpDataSphere():
    def __init__(self):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()

    def domain(self):
        return self.surface

    def time_mesh(self, t0, t1, NT, timeline='uniform'):
        if timeline is 'uniform':
            return UniformTimeLine(t0, t1, NT)
        elif timeline is 'chebyshev':
            return ChebyshevTimeLine(t0, t1, NT)

    def init_mesh(self, n=4):
        mesh = self.surface.init_mesh()
        mesh.uniform_refine(n, surface=self.surface)
        return mesh

    def solution(self, p, t):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.exp(x+y+z)*np.exp(-t)
        return u

    def source(self, p, t):
        """ The right hand side of parabolic equation
        INPUT:
            p: array object, N*3
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r = x**2 + y**2 + z**2
        t1 = -3*np.exp(x+y+z)*np.exp(-t)
        t2 = 2*np.exp(x+y+z)*np.exp(-t)*(x+y+z+x*y+x*z+y*z)

        rhs = t1 + t2/r
        return rhs
    
    def gradient(self, p):
        """ The Gradu of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi

        t1 = np.exp(x+y+z)*np.exp(-1)
        t2 = np.exp(x+y+z)*np.exp(-1)
        t3 = np.exp(x+y+z)*np.exp(-1)

        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., 0] = t1
        grad[..., 1] = t2
        grad[..., 2] = t3
        return grad  



