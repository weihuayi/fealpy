from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import IntervalMesher

class Exp0001(IntervalMesher):
    """
    1D Poisson problem:

        -u''(x) = f(x),  x in (0, 1)
         u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(πx)

    The corresponding source term is:

        f(x) = π²·sin(πx)

    Dirichlet boundary conditions are applied at both ends of the interval.
    """
    def __init__(self, rho = 1.0, mu = 1.0):
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = mu
        self.rho = rho
        super().__init__(box=self.box)

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return self.interval
    
    def init_mesh(self, nx = 8, ny = 8):
        pass
    
    @cartesian
    def velocity(self):
        pass
    
    @cartesian
    def pressure(self):
        pass
    
    @cartesian
    def source(self):
        pass

    @cartesian
    def is_velocity_boundary(self):
        pass

    @cartesian
    def is_pressure_boundary(self):
        pass

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

