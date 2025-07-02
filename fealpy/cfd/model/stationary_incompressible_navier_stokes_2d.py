from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from typing import Union, Callable, Dict
from fealpy.mesh import TriangleMesh
import sympy as sp
from fealpy.decorator import variantmethod


CoefType = Union[int, float, Callable]


class StationaryNSLFEMPolynomialPDE:
    def __init__(self):
        self.eps = 1e-10
        self.rho = 1.0
        self.mu = 1.0
        self.mesh = self.set_mesh()
    
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape)
        val[..., 0] = 10*x**2*(x-1)**2*y*(y-1)*(2*y-1)
        val[..., 1] = -10*x*(x-1)*(2*x-1)*y**2*(y-1)**2
        return val
    
    def domain(self):
        return [0, 1, 0, 1]

    def set_mesh(self, n=16):
        box = self.domain()
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        self.mesh = mesh
        return mesh
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 10*(2*x-1)*(2*y-1)
        return val
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape)
        val[..., 0] = -(20*x**2*(x - 1)**2*(6*y - 3) + 20*y*(y - 1)*(2*y - 1)*(x**2 + 4*x*(x - 1) + (x - 1)**2)) + 10*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(10*x**2*y*(2*x - 2)*(y - 1)*(2*y - 1) + 20*x*y*(x - 1)**2*(y - 1)*(2*y - 1)) - 10*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(20*x**2*y*(x - 1)**2*(y - 1) + 10*x**2*y*(x - 1)**2*(2*y - 1) + 10*x**2*(x - 1)**2*(y - 1)*(2*y - 1)) + 40*y - 20
        val[..., 1] = -(-20*x*(x - 1)*(2*x - 1)*(y**2 + 4*y*(y - 1) + (y - 1)**2) - 20*y**2*(6*x - 3)*(y - 1)**2) + 10*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(-20*x*y**2*(x - 1)*(y - 1)**2 - 10*x*y**2*(2*x - 1)*(y - 1)**2 - 10*y**2*(x - 1)*(2*x - 1)*(y - 1)**2) - 10*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(-10*x*y**2*(x - 1)*(2*x - 1)*(2*y - 2) - 20*x*y*(x - 1)*(2*x - 1)*(y - 1)**2) + 40*x - 20
        return val   

    @cartesian
    def is_p_boundary(self, p):
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result

    @cartesian
    def is_u_boundary(self, p):
        result = bm.ones_like(p[..., 0], dtype=bm.bool)
        return result
    


class FromSympy:
    def __init__(self, rho=1.0, mu=1.0) -> None:
        self.eps = 1e-10
        self.box = [0, 1, 0, 1]
        self.mu = mu
        self.rho = rho
        self.x, self.y = sp.symbols('x, y')
        self.select_pde["sin_sin"]()        
        self.mesh = self.set_mesh()
    
    @variantmethod("sin_sin")
    def select_pde(self):
        x, y = sp.symbols('x, y')
        self.u1 = sp.sin(3 * sp.pi * x) ** 2 * sp.sin(6 * sp.pi * y)
        self.u2 = -sp.sin(3 * sp.pi * y) ** 2 * sp.sin(6 * sp.pi * x)
        self.p = x ** 2 - y ** 2
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)

    @select_pde.register("Polynomial")
    def select_pde(self):
        x, y = sp.symbols('x, y')
        self.u1 = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1)
        self.u2 = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2
        self.p = 10 * (2 * x - 1) * (2 * y - 1)
        self._init_expr(self.u1, self.u2, self.p, self.mu, self.rho)

    @select_pde.register("Custom")
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

        convection = sp.Matrix([u1 * gradu1x + u2 * gradu1y, u1 * gradu2x + u2 * gradu2y])
        diffusion = sp.Matrix([sp.diff(gradu1x, x) + sp.diff(gradu1y, y), sp.diff(gradu2x, x) + sp.diff(gradu2y, y)])
        gradp = sp.Matrix([p.diff(x), p.diff(y)])
        force = - mu*diffusion + rho*convection + gradp
        
        self.u = sp.lambdify([x, y], u, 'numpy')
        self.p = sp.lambdify([x, y], p, 'numpy')
        self.force = sp.lambdify([x, y], force, 'numpy')


    def domain(self):
        return self.box
    
    def set_mesh(self, n=16):
        box = self.box
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        self.mesh = mesh
        return mesh
    
    @cartesian
    def is_p_boundary(self, p):
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result
    
    @cartesian
    def is_u_boundary(self, p):
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
    
