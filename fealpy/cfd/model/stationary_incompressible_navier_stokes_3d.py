from typing import Union, Callable, Dict

import sympy as sp

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod

from fealpy.mesher.box_mesher import BoxMesher3d 

CoefType = Union[int, float, Callable]

class FromSympy(BoxMesher3d):
    def __init__(self, rho=1.0, mu=1.0) -> None:
        self.eps = 1e-10
        self.box = [0, 1, 0, 1, 0, 1]
        self.mu = mu
        self.rho = rho
        self.x, self.y, self.z = sp.symbols('x, y, z')
        self.select_pde["sinsin"]()        
        self.mesh = self.init_mesh()
    
    @variantmethod("poly3d")
    def select_pde(self):
        x, y, z = self.x, self.y, self.z
        self.u1 = x**2*(x-1)**2*(2*y*(y-1)*(2*y-1)*z**2*(z-1)**2 - y**2*(y-1)**2*2*z*(z-1)*(2*z-1))
        self.u2 = y**2*(y-1)**2*(2*z*(z-1)*(2*z-1)*x**2*(x-1)**2 - z**2*(z-1)**2*2*x*(x-1)*(2*x-1))
        self.u3 = z**2*(z-1)**2*(2*x*(x-1)*(2*x-1)*y**2*(y-1)**2 - x**2*(x-1)**2*2*y*(y-1)*(2*y-1))
        self.p = (2*x-1)*(2*y-1)*(2*z-1)
        self._init_expr(self.u1, self.u2, self.u3, self.p, self.mu, self.rho)

    @select_pde.register("custom")
    def select_pde(self, u1, u2, u3, p, mu, rho):
        self.u1 = u1
        self.u2 = u2
        self.p = p
        self.mu = mu
        self.rho = rho
        self._init_expr(self.u1, self.u2, self.u3, self.p, self.mu, self.rho)

    def _init_expr(self, u1, u2, u3, p, mu, rho):
        x, y, z = self.x, self.y, self.z
        u = sp.Matrix([u1, u2, u3])
        gradu1x = u1.diff(x)
        gradu1y = u1.diff(y)
        gradu1z = u1.diff(z)
        gradu2x = u2.diff(x)
        gradu2y = u2.diff(y)
        gradu2z = u2.diff(z)
        gradu3x = u3.diff(x)
        gradu3y = u3.diff(y)
        gradu3z = u3.diff(z)

        # 不可压缩性
        assert sp.simplify(gradu1x + gradu2y + gradu3z) == 0  

        convection = sp.Matrix([u1 * gradu1x + u2 * gradu1y + u3 * gradu1z, u1 * gradu2x + u2 * gradu2y + u3 * gradu2z, u1 * gradu3x + u2 * gradu3y + u3 * gradu3z])
        diffusion = sp.Matrix([sp.diff(gradu1x, x) + sp.diff(gradu1y, y) + sp.diff(gradu1z, z), sp.diff(gradu2x, x) + sp.diff(gradu2y, y) + sp.diff(gradu2z, z), sp.diff(gradu3x, x) + sp.diff(gradu3y, y) + sp.diff(gradu3z, z)])
        gradp = sp.Matrix([p.diff(x), p.diff(y), p.diff(z)])
        force = - mu*diffusion + rho*convection + gradp
        
        self.u = sp.lambdify([x, y, z], u, 'numpy')
        self.p = sp.lambdify([x, y, z], p, 'numpy')
        self.force = sp.lambdify([x, y, z], force, 'numpy')


    def domain(self):
        return self.box
    
    @variantmethod("tet")
    def init_mesh(self, nx=8, ny=8, nz=8):
        from fealpy.mesh import TetrahedronMesh
        box = self.box
        mesh = TetrahedronMesh.from_box(box, nx=nx, ny=ny, nz = nz)
        self.mesh = mesh
        return mesh
    
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
        z = p[..., 2] 
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.force(x, y, z)[0])
        result[..., 1] = bm.array(self.force(x, y, z)[1])
        result[..., 2] = bm.array(self.force(x, y, z)[2])
        return result
    
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.u(x, y, z)[0])
        result[..., 1] = bm.array(self.u(x, y, z)[1])
        result[..., 2] = bm.array(self.u(x, y, z)[2])
        return result
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return bm.array(self.p(x, y, z))
    
