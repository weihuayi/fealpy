from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

from ....simulation.time import UniformTimeLine
import sympy as sp

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
        pass

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
     
    @cartesian
    def velocity(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 2* bm.pi *bm.sin(t) * bm.sin(bm.pi*x)**2 * bm.sin(bm.pi*y) * bm.cos(bm.pi*y)
        result[..., 1] = -2* bm.pi *bm.sin(t) * bm.sin(bm.pi*x) * bm.cos(bm.pi*x) * bm.sin(bm.pi*y)**2
        return result
     
    @cartesian
    def pressure(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return 20*bm.sin(t)*(x**2*y-1/6)
    
    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        cos = bm.cos
        pi = bm.pi
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 40*x*y*sin(t) - 2.0*pi*(-2*pi**2*sin(t)*sin(pi*x)**2*sin(pi*y)**2 + 2*pi**2*sin(t)*sin(pi*x)**2*cos(pi*y)**2)*sin(t)*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + 8.0*pi**3*sin(t)**2*sin(pi*x)**3*sin(pi*y)**2*cos(pi*x)*cos(pi*y)**2 + 12.0*pi**3*sin(t)*sin(pi*x)**2*sin(pi*y)*cos(pi*y) - 4.0*pi**3*sin(t)*sin(pi*y)*cos(pi*x)**2*cos(pi*y) + 2.0*pi*sin(pi*x)**2*sin(pi*y)*cos(t)*cos(pi*y)
        result[..., 1] = 20*x**2*sin(t) + 2.0*pi*(2*pi**2*sin(t)*sin(pi*x)**2*sin(pi*y)**2 - 2*pi**2*sin(t)*sin(pi*y)**2*cos(pi*x)**2)*sin(t)*sin(pi*x)**2*sin(pi*y)*cos(pi*y) + 8.0*pi**3*sin(t)**2*sin(pi*x)**2*sin(pi*y)**3*cos(pi*x)**2*cos(pi*y) - 12.0*pi**3*sin(t)*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + 4.0*pi**3*sin(t)*sin(pi*x)*cos(pi*x)*cos(pi*y)**2 - 2.0*pi*sin(pi*x)*sin(pi*y)**2*cos(t)*cos(pi*x)
        return result

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return 0

    @cartesian
    def velocity_dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        '''Compute Dirichlet boundary condition for velocity.'''
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        '''Compute Dirichlet boundary condition for pressure.'''
        x = p[..., 0]
        y = p[..., 1]
        return self.pressure(p, t)


