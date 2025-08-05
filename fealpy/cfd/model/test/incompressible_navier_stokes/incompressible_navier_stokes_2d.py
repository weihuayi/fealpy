import sympy as sp
from typing import Union, Callable, Dict

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from fealpy.mesh import TriangleMesh
from fealpy.mesher.box_mesher import BoxMesher2d

from ....simulation.time import UniformTimeLine
CoefType = Union[int, float, Callable]

class FromSympy(BoxMesher2d):
    def __init__(self, rho=1.0, mu=1.0) -> None:
        self.eps = 1e-10
        self.box = [0, 1, 0, 1]
        self.mu = mu
        self.rho = rho
        self.x, self.y, self.t = sp.symbols('x, y, t')
        self.select_pde["channel"]()        
    
    @variantmethod("channel")
    def select_pde(self):
        x, y, t = self.x, self.y, self.t
        self.u1 =  4 * y * (1-y)
        self.u2 = sp.sympify(0)
        self.p = 8 * (1-x)
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)

    @select_pde.register("polycos")
    def select_pde(self):
        x, y, t = self.x, self.y, self.t
        self.u1 = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * sp.cos(t)
        self.u2 = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * sp.cos(t)
        self.p = 10 * (2 * x - 1) * (2 * y - 1) * sp.cos(t)
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)

    @select_pde.register("poly2d")
    def select_pde(self):
        x, y, t = self.x, self.y, self.t
        self.u1 = 0.05 * sp.exp(-t) * x **2 * (x-1)**2 * (4 * y**3 - 6 * y**2 + 2*y)
        self.u2 = -0.05 * sp.exp(-t) * (4 * x**3 - 6 * x**2 + 2 * x) * y**2 * (y - 1)**2
        self.p = 0.05 * sp.exp(-t) * (x**2 + y**2 - 2/3)
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)
    
    @select_pde.register("sinsincos")
    def select_pde(self):
        x, y, t = sp.symbols('x, y, t')
        self.u1 = 2* sp.pi *sp.sin(t) * sp.sin(sp.pi*x)**2 * sp.sin(sp.pi*y) * sp.cos(sp.pi*y)
        self.u2 = -2* sp.pi *sp.sin(t) * sp.sin(sp.pi*x) * sp.cos(sp.pi*x) * sp.sin(sp.pi*y)**2
        self.p = 20*sp.sin(t)*(x**2*y-1/6)
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)


    @select_pde.register("custom")
    def select_pde(self, u1, u2, p, mu, rho):
        self.u1 = u1
        self.u2 = u2
        self.p = p
        self.mu = mu
        self.rho = rho
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)

    def _init_expr(self, u1, u2, p, mu, rho):
        x, y, t = self.x, self.y, self.t
        
        gradu1t = u1.diff(t)
        gradu2t = u2.diff(t)
        gradu1x = u1.diff(x)
        gradu1y = u1.diff(y)
        gradu2x = u2.diff(x)
        gradu2y = u2.diff(y)

        # 不可压缩性
        assert sp.simplify(gradu1x + gradu2y) == 0  

        time_derivative1 = gradu1t
        time_derivative2 = gradu2t
        convection1 = u1 * gradu1x + u2 * gradu1y
        convection2 =  u1 * gradu2x + u2 * gradu2y
        diffusion1 = sp.diff(gradu1x, x) + sp.diff(gradu1y, y)
        diffusion2 = sp.diff(gradu2x, x) + sp.diff(gradu2y, y)
        gradpx = p.diff(x)
        gradpy = p.diff(y)
        force1 = - mu*diffusion1 + rho*convection1 + gradpx + rho*time_derivative1
        force2 = - mu*diffusion2 + rho*convection2 + gradpy + rho*time_derivative2

        self.u1 = sp.lambdify((x, y, t), u1, 'numpy')
        self.u2 = sp.lambdify((x, y, t), u2, 'numpy')
        self.p = sp.lambdify((x, y, t), p, 'numpy')
        self.fx = sp.lambdify((x, y, t), force1, 'numpy')
        self.fy = sp.lambdify((x, y, t),force2, 'numpy')

    '''
    @cartesian
    def is_pressure_boundary(self, p):
        return bm.zeros_like(p[...,1],dtype=bm.bool)
        #return None
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    ''' 
    @cartesian
    def is_pressure_boundary(self, p):
        tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
        tag_right = bm.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right
    
    @cartesian
    def is_velocity_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down
    
    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.fx(x, y, t))
        result[..., 1] = bm.array(self.fy(x, y, t))
        return result
    
    @cartesian
    def velocity(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.u1(x, y, t))
        result[..., 1] = bm.array(self.u2(x, y, t))
        return result
    
    
    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        return bm.array(self.p(x, y, t))

    velocity_dirichlet = velocity
    pressure_dirichlet = pressure 
