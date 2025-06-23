from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
import sympy as sp

class AcCircleData2D:
    """
    2D Allen-Cahn phase field:

        -\phi_t + (u \cdot \nabla)\phi = \gamma(\Delta \phi - f(\phi))          (x, y) \in \Omega, t > 0
        \phi(x, y, 0) =   -tanh((\sqrt{x^2 + y^2} - r_0)/\eta)                  (x, y) \in \Omega
        where \Omega = \{(x, y) | x^2 + y^2 < r_0^2\} is a circle of radius r_0 centered at the origin.
    
    Exact solution:
        Have no exact solution, but we can use the initial condition as a reference.

    """
    def __init__(self):
        self.box = [-1.0, 1.0, -1.0, 1.0]
        self.x, self.y, self.t = sp.symbols("x y t")
        self.r0 = 100/128
        self.gamma = 6.10351e-05
        self.eta = 0.0078
        self.u = 0  # velocity field, not used in this example
        self.area = 4.0

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box  

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        r0 = self.r0
        eta = self.eta
        return -bm.tanh((r - r0) / eta)