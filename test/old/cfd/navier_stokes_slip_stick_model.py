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

class Poisuille:
    """
    [0, 1]^2
    u(x, y) = (4y(1-y), 0)
    p = 8(1-x)
    """
    def __init__(self,eps=1e-12):
        self.eps = eps
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 4*y*(1-y)
        return value

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def source(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def is_p_boundary(self,p):
        return (np.abs(p[..., 0]) < self.eps) | (np.abs(p[..., 0] - 1.0) < self.eps)
      
    @cartesian
    def is_wall_boundary(self,p):
        return (np.abs(p[..., 1]) < self.eps) | (np.abs(p[..., 1] - 1.0) < self.eps)

    @cartesian
    def p_dirichlet(self, p):
        return self.pressure(p)
    
    @cartesian
    def u_dirichlet(self, p):
        return self.velocity(p)


class FlowPastCylinder:
    """
    [0, 1]^2
    u(x, y) = (4y(1-y), 0)
    p = 8(1-x)
    """
    def __init__(self,eps=1e-12):
        self.eps = eps
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 4*y*(1-y)
        return value

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def source(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def is_outflow_boundary(self,p):
        return np.abs(p[..., 0] - 1.0) < self.eps
    
    @cartesian
    def is_inflow_boundary(self,p):
        return np.abs(p[..., 0]) < self.eps
      
    @cartesian
    def is_wall_boundary(self,p):
        return (np.abs(p[..., 1]) < self.eps) | (np.abs(p[..., 1] - 1.0) < self.eps)

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
        return self.pressure(p)
    
    @cartesian
    def u_dirichlet(self, p):
        return self.velocity(p)
