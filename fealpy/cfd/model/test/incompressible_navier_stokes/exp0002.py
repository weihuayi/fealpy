from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

from fealpy.cfd.simulation.time import UniformTimeLine
import sympy as sp

class Exp0002(BoxMesher2d):
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
        result[..., 0] = 0.05 * bm.exp(-t) * x **2 * (x-1)**2 * (4 * y**3 - 6 * y**2 + 2*y)
        result[..., 1] = -0.05 * bm.exp(-t) * (4 * x**3 - 6 * x**2 + 2 * x) * y**2 * (y - 1)**2
        return result
    
    @cartesian
    def pressure(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return 0.05 * bm.exp(-t) * (x**2 + y**2 - 2/3)
    
    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = -0.0025*x**2*y**2*(x - 1)**2*(y - 1)**2*(4*x**3 - 6*x**2 + 2*x)*(12*y**2 - 12*y + 2)*bm.exp(-2*t) - 0.05*x**2*(x - 1)**2*(24*y - 12)*bm.exp(-t) + 0.05*x**2*(x - 1)**2*(0.05*x**2*(2*x - 2)*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t) + 0.1*x*(x - 1)**2*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t))*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t) - 0.05*x**2*(x - 1)**2*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t) - 0.1*x**2*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t) - 0.2*x*(2*x - 2)*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t) + 0.1*x*bm.exp(-t) - 0.1*(x - 1)**2*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-t)
        result[..., 1] = -0.0025*x**2*y**2*(x - 1)**2*(y - 1)**2*(12*x**2 - 12*x + 2)*(4*y**3 - 6*y**2 + 2*y)*bm.exp(-2*t) + 0.05*y**2*(24*x - 12)*(y - 1)**2*bm.exp(-t) - 0.05*y**2*(y - 1)**2*(-0.05*y**2*(2*y - 2)*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t) - 0.1*y*(y - 1)**2*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t))*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t) + 0.05*y**2*(y - 1)**2*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t) + 0.1*y**2*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t) + 0.2*y*(2*y - 2)*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t) + 0.1*y*bm.exp(-t) + 0.1*(y - 1)**2*(4*x**3 - 6*x**2 + 2*x)*bm.exp(-t)
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
        return self.velocity(p, t)
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        '''Compute Dirichlet boundary condition for pressure.'''
        x = p[..., 0]
        y = p[..., 1]
        return self.pressure(p, t)
