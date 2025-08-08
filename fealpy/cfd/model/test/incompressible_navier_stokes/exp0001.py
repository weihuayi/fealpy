from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

from ....simulation.time import UniformTimeLine
import sympy as sp

class Exp0001(BoxMesher2d):
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.mesh = self.init_mesh(nx=options.get('nx', 8), ny=options.get('ny', 8))
        # self._init_expr()
        super().__init__(box=self.box)

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the PDE configuration."""
        pass

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def set_mesh(self, nx, ny):
        mesh = super().init_mesh['uniform_tri'](nx=nx, ny=ny)
        self.mesh = mesh 
        return mesh
     
    @cartesian
    def velocity(self, p: TensorLike, t) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * bm.cos(t)
        result[..., 1] = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * bm.cos(t)
        return result
    
    @cartesian
    def velocity_0(self, p):
        return self.velocity(p, self.t0)

    @cartesian
    def pressure_0(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return self.pressure(p, self.t0)
    
    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        return 10 * (2 * x - 1) * (2 * y - 1) * bm.cos(t)
    
    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 10.0*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(10*x**2*y*(2*x - 2)*(y - 1)*(2*y - 1)*bm.cos(t) + 20*x*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t))*bm.cos(t) - 10*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.sin(t) - 40.0*x**2*y*(x - 1)**2*bm.cos(t) - 20.0*x**2*y*(y - 1)*(2*y - 1)*bm.cos(t) - 40.0*x**2*(x - 1)**2*(y - 1)*bm.cos(t) - 20.0*x**2*(x - 1)**2*(2*y - 1)*bm.cos(t) - 10.0*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(20*x**2*y*(x - 1)**2*(y - 1)*bm.cos(t) + 10*x**2*y*(x - 1)**2*(2*y - 1)*bm.cos(t) + 10*x**2*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t))*bm.cos(t) - 40.0*x*y*(2*x - 2)*(y - 1)*(2*y - 1)*bm.cos(t) - 20.0*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t) + 20*(2*y - 1)*bm.cos(t)
        result[..., 1] = 10.0*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(-20*x*y**2*(x - 1)*(y - 1)**2*bm.cos(t) - 10*x*y**2*(2*x - 1)*(y - 1)**2*bm.cos(t) - 10*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t))*bm.cos(t) - 10.0*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(-10*x*y**2*(x - 1)*(2*x - 1)*(2*y - 2)*bm.cos(t) - 20*x*y*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t))*bm.cos(t) + 10*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*bm.sin(t) + 20.0*x*y**2*(x - 1)*(2*x - 1)*bm.cos(t) + 40.0*x*y**2*(y - 1)**2*bm.cos(t) + 40.0*x*y*(x - 1)*(2*x - 1)*(2*y - 2)*bm.cos(t) + 20.0*x*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t) + 40.0*y**2*(x - 1)*(y - 1)**2*bm.cos(t) + 20.0*y**2*(2*x - 1)*(y - 1)**2*bm.cos(t) + 2*(20*x - 10)*bm.cos(t)
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # result = bm.ones_like(p[..., 0], dtype=bm.bool)
        # return result
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return 0

    @cartesian
    def velocity_dirichlet(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        return None

    def _init_expr(self):
        x, y, t = sp.symbols('x, y, t')
        u1 = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * sp.cos(t)
        u2 = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * sp.cos(t)
        p = 10 * (2 * x - 1) * (2 * y - 1) * sp.cos(t)
        mu = self.mu
        rho = self.rho
        u = sp.Matrix([u1, u2])
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
        force11 = - mu*diffusion1 + rho*convection1 + gradpx + time_derivative1
        force22 = - mu*diffusion2 + rho*convection2 + gradpy + time_derivative2
        print(force11)
        print(force22)

        self.u1 = sp.lambdify((x, y, t), u1, 'numpy')
        self.u2 = sp.lambdify((x, y, t), u2, 'numpy')
        self.p = sp.lambdify((x, y, t), p, 'numpy')
        self.fx = sp.lambdify((x, y, t), force11, 'numpy')
        self.fy = sp.lambdify((x, y, t),force22, 'numpy')
