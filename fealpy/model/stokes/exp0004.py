from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0004(BoxMesher2d):
    
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.mesh = self.init_mesh['uniform_tri'](nx=options.get('nx', 8), ny=options.get('ny', 8))
        super().__init__(box=self.box)

 
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 1000 * x**2 * (1 - x)**4 * y**2 * (1 - y) * (3 - 5*y)
        result[..., 1] = 1000 * (-2) * x * (1 - x)**3 * (1 - 3*x) * y**3 * (1 - y)**2
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        cos = bm.cos
        sin = bm.sin
        return pi**2 * (x*y**3 * cos(2*pi*x**2*y) - x**2*y * sin(2*pi*x*y)) + 1/8

    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos

        term1 = -10000.0 * x**2 * y**2 * (1 - x)**4
        term2 = -12000.0 * x**2 * y**2 * (1 - x)**2 * (1 - y) * (3 - 5*y)
        term3 =  20000.0 * x**2 * y * (1 - x)**4 * (1 - y)
        term4 =   4000.0 * x**2 * y * (1 - x)**4 * (3 - 5*y)
        term5 =  -2000.0 * x**2 * (1 - x)**4 * (1 - y) * (3 - 5*y)
        term6 =  16000.0 * x * y**2 * (1 - x)**3 * (1 - y) * (3 - 5*y)
        term7 =  -2000.0 * y**2 * (1 - x)**4 * (1 - y) * (3 - 5*y)

        trig_x = pi**2 * (
            -4*pi*x**2*y**4*sin(2*pi*x**2*y)
            - 2*pi*x**2*y**2*cos(2*pi*x*y)
            - 2*x*y*sin(2*pi*x*y)
            + y**3*cos(2*pi*x**2*y)
        )

        result = bm.set_at(result, (..., 0), term1 + term2 + term3 + term4 + term5 + term6 + term7 + trig_x)

        term1 =  4000.0 * x * y**3 * (1 - 3*x) * (1 - x)**3
        term2 = -6000.0 * x * y**3 * (1 - 3*x) * (1 - y)**2 * (2*x - 2)
        term3 = 36000.0 * x * y**3 * (1 - x)**2 * (1 - y)**2
        term4 = 12000.0 * x * y**2 * (1 - 3*x) * (1 - x)**3 * (2*y - 2)
        term5 = 12000.0 * x * y * (1 - 3*x) * (1 - x)**3 * (1 - y)**2
        term6 = -12000.0 * y**3 * (1 - 3*x) * (1 - x)**2 * (1 - y)**2
        term7 = -12000.0 * y**3 * (1 - x)**3 * (1 - y)**2

        trig_y = pi**2 * (
            -2*pi*x**3*y**3*sin(2*pi*x**2*y)
            - 2*pi*x**3*y*cos(2*pi*x*y)
            - x**2*sin(2*pi*x*y)
            + 3*x*y**2*cos(2*pi*x**2*y)
        )

        result = bm.set_at(result, (..., 1), term1 + term2 + term3 + term4 + term5 + term6 + term7 + trig_y)

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
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        return self.velocity(p)
    
    @cartesian
    def dirichlet_pressure(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        return self.pressure(p)
    

    # === split-component functions ===
    @cartesian
    def pressure_integral_target(self) -> float:
        return 0.0

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        return self.velocity_u(p)

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        return self.velocity_v(p)

    @cartesian
    def source_u(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 1]

    @cartesian
    def velocity_u(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return 1000 * x**2 * (1 - x)**4 * y**2 * (1 - y) * (3 - 5*y)

    @cartesian
    def velocity_v(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return 1000 * (-2) * x * (1 - x)**3 * (1 - 3*x) * y**3 * (1 - y)**2