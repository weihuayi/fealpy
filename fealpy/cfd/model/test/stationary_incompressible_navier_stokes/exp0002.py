from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d
import sympy as sp

class Exp0002(BoxMesher2d):
    
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.mesh = self.init_mesh(nx=options.get('nx', 8), ny=options.get('ny', 8))
        self._init_expr()
        super().__init__(box=self.box)

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the PDE configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  problem            : 2D stationary incompressible Navier-Stokes\n"
        s += f"  domain             : {self.box}\n"
        s += f"  mesh size          : nx = {self.nx}, ny = {self.ny}\n"
        s += f"  density (ρ)        : {self.rho}\n"
        s += f"  viscosity (μ)      : {self.mu}\n"
        s += f"  exact_velocity_x   : u_1(x, y) = 10·x²·(x - 1)²·y·(y - 1)·(2y - 1)\n"
        s += f"  exact_velocity_y   : u_2(x, y) = -10·x·(x - 1)·(2x - 1)·y²·(y - 1)²\n"
        s += f"  exact_pressure     : p(x, y) = 10·(2x - 1)·(2y - 1)\n"
        s += f")"
        return s

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def init_mesh(self, nx, ny):
        mesh = super().init_mesh['uniform_tri'](nx=nx, ny=ny)
        self.nx = nx
        self.ny = ny
        return mesh
    
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.u(x, y)[0])
        result[..., 1] = bm.array(self.u(x, y)[1])
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return bm.array(self.p(x, y))
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(self.force(x, y)[0])
        result[..., 1] = bm.array(self.force(x, y)[1])
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # result = bm.ones_like(p[..., 0], dtype=bm.bool)
        # return result
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result

    @cartesian
    def velocity_gradient(self, p: TensorLike) -> TensorLike:
        pass
    
    @cartesian
    def pressure_gradient(self):
        pass

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        pass
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        pass

    def _init_expr(self):
        x, y = sp.symbols('x, y')
        u1 = 0.1/(2 * sp.pi) * sp.exp(0.1*y)/(sp.exp(0.1)-1)*sp.sin(2*sp.pi*(sp.exp(0.1*y)-1)/(sp.exp(0.1)-1))*(1-sp.cos(2*sp.pi*(sp.exp(3*x)-1)/(sp.exp(3)-1)))
        u2 = -3/(2 * sp.pi) * sp.exp(3*x)/(sp.exp(3)-1)*sp.sin(2*sp.pi*(sp.exp(3*x)-1)/(sp.exp(3)-1))*(1-sp.cos(2*sp.pi*(sp.exp(0.1*y)-1)/(sp.exp(0.1)-1)))
        p = 0.3*sp.exp(3*x)*sp.exp(0.1*y)/((sp.exp(3)-1)*(sp.exp(0.1)-1)) * sp.sin(2*sp.pi*(sp.exp(3*x)-1)/(sp.exp(3)-1)) * sp.sin(2*sp.pi*(sp.exp(0.1*y)-1)/(sp.exp(0.1)-1))* (1-sp.sin(2*sp.pi*(sp.exp(3*x)-1)/(sp.exp(3)-1)))
        
        mu = self.mu
        rho = self.rho
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

