from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from typing import Union, Callable, Dict
from fealpy.mesh import TriangleMesh, QuadrangleMesh
import sympy as sp
from fealpy.mesher.box_mesher import BoxMesher2d 


CoefType = Union[int, float, Callable]


class FromSympy(BoxMesher2d):
    def __init__(self, rho=1.0, mu=1.0) -> None:
        self.eps = 1e-10
        self.box = [0, 1, 0, 1]
        self.mu = mu
        self.rho = rho
        self.x, self.y = sp.symbols('x, y')
        self.select_pde["poly2d"]()        
    
    @variantmethod("poly2d")
    def select_pde(self):
        x, y = sp.symbols('x, y')
        self.u1 = 4*(2*y-1)*x*(1-x)
        self.u2 = -4*(2*x-1)*y*(1-y)
        self.p = 3*(x**3 + y**3 -sp.Rational(1,2))
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
        x, y = self.x, self.y
        u = sp.Matrix([u1, u2])
        gradu1x = u1.diff(x)
        gradu1y = u1.diff(y)
        gradu2x = u2.diff(x)
        gradu2y = u2.diff(y)

        # 不可压缩性
        assert sp.simplify(gradu1x + gradu2y) == 0  

        diffusion = sp.Matrix([sp.diff(gradu1x, x) + sp.diff(gradu1y, y), sp.diff(gradu2x, x) + sp.diff(gradu2y, y)])
        gradp = sp.Matrix([p.diff(x), p.diff(y)])
        force = - mu*diffusion + gradp
        print(f"force_0: {force[0]}")
        print(f"force_1: {force[1]}")
        
        self.u = sp.lambdify([x, y], u, 'numpy')
        self.p = sp.lambdify([x, y], p, 'numpy')
        self.force = sp.lambdify([x, y], force, 'numpy')


    def domain(self):
        return self.box
   
    @cartesian
    def is_pressure_boundary(self, p):
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.force(x, y)[0])
        result[..., 1] = bm.array(self.force(x, y)[1])
        return result
    
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.u(x, y)[0])
        result[..., 1] = bm.array(self.u(x, y)[1])
        return result
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.array(self.p(x, y))
    
    @cartesian
    def velocity_dirichlet(self, p):
        return self.velocity(p)
    
    @cartesian
    def pressure_dirichlet(self, p):
        return self.pressure(p)


pde = FromSympy(rho=1.0, mu=1.0)
pde.select_pde["poly2d"]()