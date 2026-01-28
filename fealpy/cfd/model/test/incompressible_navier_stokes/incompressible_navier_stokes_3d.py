from typing import Union, Callable, Dict

import sympy as sp

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,variantmethod
from typing import Union, Callable, Dict
import sympy as sp
from fealpy.mesher.box_mesher import BoxMesher3d 

from fealpy.mesher.box_mesher import BoxMesher3d 

CoefType = Union[int, float, Callable]

class FromSympy(BoxMesher3d):
    def __init__(self, rho=1.0, mu=1.0) -> None:
        self.eps = 1e-10
        self.box = [0, 1, 0, 1, 0, 1]
        self.mu = mu
        self.rho = rho
        self.x, self.y, self.z, self.t= sp.symbols('x, y, z, t')       
        self.mesh = self.init_mesh()

    def _init_expr(self, u1, u2, u3, p, mu, rho):
        x, y, z, t = self.x, self.y, self.z, self.t
        u = sp.Matrix([u1, u2, u3])

        gradu1t = u1.diff(t)
        gradu2t = u2.diff(t)
        gradu3t = u3.diff(t)
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

        time_derivative = sp.Matrix([gradu1t, gradu2t, gradu3t])
        convection = sp.Matrix([u1 * gradu1x + u2 * gradu1y + u3 * gradu1z, u1 * gradu2x + u2 * gradu2y + u3 * gradu2z, u1 * gradu3x + u2 * gradu3y + u3 * gradu3z])
        diffusion = sp.Matrix([sp.diff(gradu1x, x) + sp.diff(gradu1y, y) + sp.diff(gradu1z, z), sp.diff(gradu2x, x) + sp.diff(gradu2y, y) + sp.diff(gradu2z, z), sp.diff(gradu3x, x) + sp.diff(gradu3y, y) + sp.diff(gradu3z, z)])
        gradp = sp.Matrix([p.diff(x), p.diff(y), p.diff(z)])
        force = - mu*diffusion + rho*convection + gradp + rho*time_derivative
        print("force_x =", force[0])
        print("force_y =", force[1])
        print("force_z =", force[2])
        
        self.u = sp.lambdify([x, y, z, t], u, 'numpy')
        self.p = sp.lambdify([x, y, z, t], p, 'numpy')
        self.force = sp.lambdify([x, y, z, t], force, 'numpy')


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
    


pde = FromSympy()
x, y, z, t = pde.x, pde.y, pde.z, pde.t
a = sp.pi/4
d = sp.pi/2
mu = 1.0
rho = 1.0
exp = sp.exp
sin = sp.sin
cos = sp.cos
u1 = -a*(exp(a*x)*sin(a*y + d*z)+ exp(a*z)*cos(a*x + d*y))*exp(-d*t**2)
u2 = -a*(exp(a*y)*sin(a*z + d*x)+ exp(a*x)*cos(a*y + d*z))*exp(-d*t**2)
u3 = -a*(exp(a*z)*sin(a*x + d*y)+ exp(a*y)*cos(a*z + d*x))*exp(-d*t**2)
p = (-a**2*exp(-2*d*t**2)*(exp(2*a*x) + exp(2*a*y) + exp(2*a*z))
    *(sin(a*x + d*y)*cos(a*z + d*x)*exp(a*(y+z)) 
    + sin(a*y + d*z)*cos(a*x + d*y)*exp(a*(z+x)) 
    + sin(a*z + d*x)*cos(a*y + d*z)*exp(a*(x+y))))
pde._init_expr(u1, u2, u3, p, mu, rho)