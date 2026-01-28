import sympy as sp
from sympy.utilities.lambdify import lambdify
from typing import Sequence

from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import IntervalMesher

class Exp0007(IntervalMesher):
    """
    1D Helmholtz problem with Sommerfeld radiation boundary condition:
    
        -Δv(x) - κ²v(x) = u_s(x),   x ∈ (0,1)
        ∂v/∂n - iκv = 0,    on ∂Ω  
    
    with the source term:
        u_s(x) = -0.5*exp(-300*(x-0.4)²) - 0.5*exp(-300*(x-0.6)²)

    The Robin boundary condition is applied on the boundaries.

    Note: This is a complex-valued problem.
    """
    def __init__(self, options: dict = {}):
        self.box = [0.0, 1.0] 
        super().__init__(interval=self.box)
        self.k = bm.tensor(options.get('k', 1.0))

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike):
        """Exact solution is unknown, return NotImplementedError."""
        raise NotImplementedError("The exact solution is unknown.")

    @cartesian
    def gradient(self, p: TensorLike):
        """Gradient is unknown, return NotImplementedError."""
        raise NotImplementedError("The exact gradient is unknown.")
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term u_s(x) = -0.5*exp(-300*(x-0.4)²) - 0.5*exp(-300*(x-0.6)²)"""
        x = p[..., 0]
        term1 = 0.5 * bm.exp(-300 * (x - 0.4)**2)
        term2 = 0.5 * bm.exp(-300 * (x - 0.6)**2)
        val = -term1 - term2
        return val
        
    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Robin boundary condition for Sommerfeld radiation condition.
        
        Transforms the radiation condition ∂v/∂r = iκv into Robin form:
        ∂v/∂n - iκv = 0,   on ∂Ω
        """
        val = bm.zeros_like(p)      
        return val

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        """
        Check if point is on Robin boundary (for radiation condition).
        
        Returns:
            Boolean tensor indicating Robin boundary points
        """
        x = p[..., 0]
        atol = 1e-12  # Absolute tolerance for boundary detection
        on_boundary = (bm.abs(x) < atol) | (bm.abs(x - 1.0) < atol)
        return on_boundary