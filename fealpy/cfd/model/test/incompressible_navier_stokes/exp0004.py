from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher3d

from ....simulation.time import UniformTimeLine
import sympy as sp

class Exp0004(BoxMesher3d):
    def __init__(self, options: dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        box = options.get("box", [0, 1, 0, 1, 0, 1])
        nx = options.get("nx", 4)
        ny = options.get("ny", 4)
        nz = options.get("nz", 4)
        super().__init__(box=box)
        self.mesh = self.init_mesh(nx=nx, ny=ny, nz=nz)

    @cartesian
    def velocity(self, p, t):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        pi = bm.pi
        a = pi/4
        d = pi/2
        exp = bm.exp
        sin = bm.sin
        cos = bm.cos
        value = bm.zeros(p.shape)
        value[...,0] = -a*(exp(a*x)*sin(a*y + d*z)+ exp(a*z)*cos(a*x + d*y))*exp(-d*t**2)
        value[...,1] = -a*(exp(a*y)*sin(a*z + d*x)+ exp(a*x)*cos(a*y + d*z))*exp(-d*t**2)
        value[...,2] = -a*(exp(a*z)*sin(a*x + d*y)+ exp(a*y)*cos(a*z + d*x))*exp(-d*t**2)
        return value
    
    @cartesian
    def pressure(self, p, t):
        z = p[..., 2]
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        a = pi/4
        d = pi/2
        exp = bm.exp
        sin = bm.sin
        cos = bm.cos
        val = bm.zeros(z.shape)
        val = (-a**2*exp(-2*d*t**2)*(exp(2*a*x) + exp(2*a*y) + exp(2*a*z))
            *(sin(a*x + d*y)*cos(a*z + d*x)*exp(a*(y+z)) 
            + sin(a*y + d*z)*cos(a*x + d*y)*exp(a*(z+x)) 
            + sin(a*z + d*x)*cos(a*y + d*z)*exp(a*(x+y))))
        return val
    
    @cartesian
    def source(self, p, t):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        f = bm.zeros(p.shape)
        pi = bm.pi
        exp = bm.exp
        sin = bm.sin
        cos = bm.cos
        f[..., 0] = (0.25*pi**2*t*(exp(pi*x/4)*sin(pi*y/4 + pi*z/2) + exp(pi*z/4)*cos(pi*x/4 
                    + pi*y/2))*exp(-pi*t**2/2) + 0.0625*pi**2*(exp(pi*x/4)*sin(pi*y/4 + pi*z/2) 
                    + exp(pi*z/4)*cos(pi*x/4 + pi*y/2))*(pi*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/4 
                    - pi*exp(pi*z/4)*sin(pi*x/4 + pi*y/2)/4)*exp(-pi*t**2) 
                    + 0.0625*pi**2*(exp(pi*x/4)*cos(pi*y/4 + pi*z/2) + exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4))*(pi*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/4 - pi*exp(pi*z/4)*sin(pi*x/4 
                    + pi*y/2)/2)*exp(-pi*t**2) + 0.0625*pi**2*(exp(pi*y/4)*cos(pi*x/2 + pi*z/4) 
                    + exp(pi*z/4)*sin(pi*x/4 + pi*y/2))*(pi*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/2 
                    + pi*exp(pi*z/4)*cos(pi*x/4 + pi*y/2)/4)*exp(-pi*t**2) 
                    + 0.25*pi*(-pi**2*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/4 
                    + pi**2*exp(pi*z/4)*cos(pi*x/4 + pi*y/2)/16)*exp(-pi*t**2/2) 
                    + 0.25*pi*(-pi**2*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/16 
                    - pi**2*exp(pi*z/4)*cos(pi*x/4 + pi*y/2)/4)*exp(-pi*t**2/2) 
                    + 0.25*pi*(pi**2*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/16 
                    - pi**2*exp(pi*z/4)*cos(pi*x/4 + pi*y/2)/16)*exp(-pi*t**2/2) 
                    - pi**3*(exp(pi*(x + y)/4)*sin(pi*x/2 + pi*z/4)*cos(pi*y/4 + pi*z/2) 
                    + exp(pi*(x + z)/4)*sin(pi*y/4 + pi*z/2)*cos(pi*x/4 + pi*y/2) 
                    + exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4))*exp(-pi*t**2)*exp(pi*x/2)/32 
                    - pi**2*(exp(pi*x/2) + exp(pi*y/2) + exp(pi*z/2))*(pi*exp(pi*(x + y)/4)*sin(pi*x/2 
                    + pi*z/4)*cos(pi*y/4 + pi*z/2)/4 + pi*exp(pi*(x + y)/4)*cos(pi*x/2 + pi*z/4)*cos(pi*y/4 
                    + pi*z/2)/2 - pi*exp(pi*(x + z)/4)*sin(pi*x/4 + pi*y/2)*sin(pi*y/4 + pi*z/2)/4 
                    + pi*exp(pi*(x + z)/4)*sin(pi*y/4 + pi*z/2)*cos(pi*x/4 + pi*y/2)/4 
                    - pi*exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*sin(pi*x/2 + pi*z/4)/2 
                    + pi*exp(pi*(y + z)/4)*cos(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4)/4)*exp(-pi*t**2)/16)
        f[..., 1] = (0.25*pi**2*t*(exp(pi*x/4)*cos(pi*y/4 + pi*z/2) + exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4))*exp(-pi*t**2/2) + 0.0625*pi**2*(exp(pi*x/4)*sin(pi*y/4 + pi*z/2) 
                    + exp(pi*z/4)*cos(pi*x/4 + pi*y/2))*(pi*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/4 
                    + pi*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/2)*exp(-pi*t**2) 
                    + 0.0625*pi**2*(exp(pi*x/4)*cos(pi*y/4 + pi*z/2) + exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4))*(-pi*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/4 + pi*exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4)/4)*exp(-pi*t**2) + 0.0625*pi**2*(exp(pi*y/4)*cos(pi*x/2 + pi*z/4) 
                    + exp(pi*z/4)*sin(pi*x/4 + pi*y/2))*(-pi*exp(pi*x/4)*sin(pi*y/4 + pi*z/2)/2 
                    + pi*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/4)*exp(-pi*t**2) 
                    + 0.25*pi*(-pi**2*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/4 - pi**2*exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4)/16)*exp(-pi*t**2/2) + 0.25*pi*(-pi**2*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/16 
                    + pi**2*exp(pi*y/4)*sin(pi*x/2 + pi*z/4)/16)*exp(-pi*t**2/2) 
                    + 0.25*pi*(pi**2*exp(pi*x/4)*cos(pi*y/4 + pi*z/2)/16 
                    - pi**2*exp(pi*y/4)*sin(pi*x/2 + pi*z/4)/4)*exp(-pi*t**2/2) 
                    - pi**3*(exp(pi*(x + y)/4)*sin(pi*x/2 + pi*z/4)*cos(pi*y/4 + pi*z/2) 
                    + exp(pi*(x + z)/4)*sin(pi*y/4 + pi*z/2)*cos(pi*x/4 + pi*y/2) 
                    + exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4))*exp(-pi*t**2)*exp(pi*y/2)/32 
                    - pi**2*(exp(pi*x/2) + exp(pi*y/2) + exp(pi*z/2))*(-pi*exp(pi*(x + y)/4)*sin(pi*x/2 
                    + pi*z/4)*sin(pi*y/4 + pi*z/2)/4 + pi*exp(pi*(x + y)/4)*sin(pi*x/2 
                    + pi*z/4)*cos(pi*y/4 + pi*z/2)/4 - pi*exp(pi*(x + z)/4)*sin(pi*x/4 + pi*y/2)*sin(pi*y/4 
                    + pi*z/2)/2 + pi*exp(pi*(x + z)/4)*cos(pi*x/4 + pi*y/2)*cos(pi*y/4 + pi*z/2)/4 
                    + pi*exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4)/4 
                    + pi*exp(pi*(y + z)/4)*cos(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4)/2)*exp(-pi*t**2)/16)
        f[..., 2] = (0.25*pi**2*t*(exp(pi*y/4)*cos(pi*x/2 + pi*z/4) + exp(pi*z/4)*sin(pi*x/4 
                    + pi*y/2))*exp(-pi*t**2/2) + 0.0625*pi**2*(exp(pi*x/4)*sin(pi*y/4 + pi*z/2) 
                    + exp(pi*z/4)*cos(pi*x/4 + pi*y/2))*(-pi*exp(pi*y/4)*sin(pi*x/2 + pi*z/4)/2 
                    + pi*exp(pi*z/4)*cos(pi*x/4 + pi*y/2)/4)*exp(-pi*t**2) 
                    + 0.0625*pi**2*(exp(pi*x/4)*cos(pi*y/4 + pi*z/2) + exp(pi*y/4)*sin(pi*x/2 
                    + pi*z/4))*(pi*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/4 + pi*exp(pi*z/4)*cos(pi*x/4 
                    + pi*y/2)/2)*exp(-pi*t**2) + 0.0625*pi**2*(exp(pi*y/4)*cos(pi*x/2 + pi*z/4) 
                    + exp(pi*z/4)*sin(pi*x/4 + pi*y/2))*(-pi*exp(pi*y/4)*sin(pi*x/2 + pi*z/4)/4 
                    + pi*exp(pi*z/4)*sin(pi*x/4 + pi*y/2)/4)*exp(-pi*t**2) 
                    + 0.25*pi*(-pi**2*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/4 - pi**2*exp(pi*z/4)*sin(pi*x/4 
                    + pi*y/2)/16)*exp(-pi*t**2/2) + 0.25*pi*(-pi**2*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/16 
                    + pi**2*exp(pi*z/4)*sin(pi*x/4 + pi*y/2)/16)*exp(-pi*t**2/2) 
                    + 0.25*pi*(pi**2*exp(pi*y/4)*cos(pi*x/2 + pi*z/4)/16 - pi**2*exp(pi*z/4)*sin(pi*x/4 
                    + pi*y/2)/4)*exp(-pi*t**2/2) - pi**3*(exp(pi*(x + y)/4)*sin(pi*x/2 + pi*z/4)*cos(pi*y/4 
                    + pi*z/2) + exp(pi*(x + z)/4)*sin(pi*y/4 + pi*z/2)*cos(pi*x/4 + pi*y/2) 
                    + exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4))*exp(-pi*t**2)*exp(pi*z/2)/32 
                    - pi**2*(exp(pi*x/2) + exp(pi*y/2) + exp(pi*z/2))*(-pi*exp(pi*(x + y)/4)*sin(pi*x/2 
                    + pi*z/4)*sin(pi*y/4 + pi*z/2)/2 + pi*exp(pi*(x + y)/4)*cos(pi*x/2 + pi*z/4)*cos(pi*y/4 
                    + pi*z/2)/4 + pi*exp(pi*(x + z)/4)*sin(pi*y/4 + pi*z/2)*cos(pi*x/4 + pi*y/2)/4 
                    + pi*exp(pi*(x + z)/4)*cos(pi*x/4 + pi*y/2)*cos(pi*y/4 + pi*z/2)/2 
                    - pi*exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*sin(pi*x/2 + pi*z/4)/4 
                    + pi*exp(pi*(y + z)/4)*sin(pi*x/4 + pi*y/2)*cos(pi*x/2 + pi*z/4)/4)*exp(-pi*t**2)/16)
        return f
    
    @cartesian
    def is_pressure_boundary(self, p = None):
        # # if p is None:
        # #     return 1
        # # tag_left = bm.abs(p[..., 2]) < self.eps
        # # tag_right = bm.abs(p[..., 2] - 1.0) < self.eps
        # # return tag_left | tag_right
        return 0
        # return 1

    @cartesian
    def is_velocity_boundary(self, p):
        # x = p[..., 0]
        # y = p[..., 1]
        # z = p[..., 2]
        # r = self.radius
        # tag = bm.abs(x**2 + y**2 - r**2) < 1e-3
        # return tag
        return None
    
    @cartesian
    def velocity_dirichlet(self, p, t):
        return self.velocity(p, t)
    
    @cartesian
    def pressure_dirichlet(self, p, t):
        return self.pressure(p, t)

