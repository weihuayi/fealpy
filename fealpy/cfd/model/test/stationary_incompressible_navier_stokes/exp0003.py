from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0003(BoxMesher2d):
    
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        super().__init__(box=self.box)
        self.mesh = self.init_mesh[options.get('init_mesh', 'uniform_tri')](nx=options.get('nx', 8), ny=options.get('ny', 8))

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the PDE configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  problem            : 2D stationary incompressible Navier-Stokes\n"
        s += f"  domain             : {self.box}\n"
        # s += f"  mesh size          : nx = {self.nx}, ny = {self.ny}\n"
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
    
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = -bm.cos(2*pi*x) * bm.sin(2*pi*y)
        result[..., 1] = bm.sin(2*pi*x) * bm.cos(2*pi*y)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return -0.25 * (bm.cos(4*pi*x) + bm.sin(4*pi*y))
    
    def pressure_integral_target(self) -> float:
        """Integral of the exact pressure over the domain."""
        return 0.0
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = -2.0*pi*sin(2*pi*x)*sin(2*pi*y)**2*cos(2*pi*x) - 2.0*pi*sin(2*pi*x)*cos(2*pi*x)*cos(2*pi*y)**2 + pi*sin(4*pi*x) - 8.0*pi**2*sin(2*pi*y)*cos(2*pi*x)
        result[..., 1] = -2.0*pi*sin(2*pi*x)**2*sin(2*pi*y)*cos(2*pi*y) + 8.0*pi**2*sin(2*pi*x)*cos(2*pi*y) - 2.0*pi*sin(2*pi*y)*cos(2*pi*x)**2*cos(2*pi*y) - pi*cos(4*pi*y)
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        if p is None:
            return 0
        result = bm.zeros_like(p[..., 0], dtype=bm.bool)
        return result

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        return self.velocity(p)
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        return self.pressure(p)
