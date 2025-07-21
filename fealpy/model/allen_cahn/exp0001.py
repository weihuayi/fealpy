from typing import Sequence
from ...decorator import cartesian,variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    2D Allen-Cahn phase-field model without advection.

    Governing equation:
        ∂φ/∂t = γ(Δφ - f(φ))        for (x, y) ∈ Ω, t > 0

    where:
        - φ(x, y, t) is the phase field,
        - f(φ) = (φ³ - φ)/η² is the nonlinear source term,
        - γ is the interface mobility (gamma),
        - η is the interface width,
        - Ω = {(x, y) | x² + y² < r₀²} is a circular domain of radius r₀ centered at the origin.

    Initial condition:
        φ(x, y, 0) = -tanh((√(x² + y²) - r₀)/η)

    Note:
        There is no known analytical solution for this PDE,
        but the initial condition can be used as a reference for validation.
    """

    def __init__(self, option: dict = {}):
        self.box = [-1, 1, -1, 1]
        super().__init__(box=self.box)
        self.u = 0
        self.r0 = bm.tensor(option.get('r0', 100/128))
        self.gam = bm.tensor(option.get('gam', 6.10351e-05))
        self.eta = bm.tensor(option.get('eta', 0.0078))
        self.area = bm.tensor(option.get('area', 4))
        
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box  
    
    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 5000.0]
    
    def nonlinear_source(self, phi):
        """Return the nonlinear source term f(φ) = 1/η^2 *(φ^3 - φ)."""
        eta = self.eta
        return (phi**3 - phi) / (eta**2)
    
    def gamma(self) -> float:
        """Return the gamma parameter in the Allen-Cahn equation."""
        return self.gam
    
    def velocity_field(self, p: TensorLike,t = 0.0) -> TensorLike:
        """Return the velocity field u."""
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(p, dtype=bm.float64)

    @cartesian
    def init_solution(self, p: TensorLike,t = 0.0) -> TensorLike:
        """
        Initialize the phase field φ(x, y, 0) as a smoothed interface centered at the origin:
            φ₀(x, y) = -tanh((r - r₀)/η), where r = √(x² + y²)
        """
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt((x)**2 + (y)**2)
        r0 = self.r0
        eta = self.eta
        return -bm.tanh((r - r0) / eta)
