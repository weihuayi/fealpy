import numpy as np

from fealpy.decorator import cartesian

class SinCosData:
    """
    [0, 1]^2
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y) 
        val[..., 1] = -cos(pi*x)*sin(pi*y) 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(y**2 + 1) - pi/4 
        return val
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*pi**2*sin(pi*x)*cos(pi*y) + pi*sin(pi*x)*cos(pi*x)
        val[..., 1] = -2*y/(y**2 + 1)**2 - 2*pi**2*sin(pi*y)*cos(pi*x) + pi*sin(pi*y)*cos(pi*x) 
        return val


    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)
