from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike

class AcCircleData2D:
    """
    2D Allen-Cahn phase field:

        -\phi_t + (u \cdot \nabla)\phi = \gamma(\Delta \phi - f(\phi))          (x, y) \in \Omega, t > 0
        \phi(x, y, 0) =   -tanh((\sqrt{x^2 + y^2} - r_0)/\eta)                  (x, y) \in \Omega
        where \Omega = \{(x, y) | x^2 + y^2 < r_0^2\} is a circle of radius r_0 centered at the origin.
    
    Exact solution:
        Have no exact solution, but we can use the initial condition as a reference.

    """
    def __init__(self,u = 0):
        self.box = [-1, 1, -1, 1]
        self.r0 = 100/128
        self.gamma = 6.10351e-05
        self.eta = 0.0078
        self.area = 4

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box  

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 5000.0]

    def velocity_field(self, p: TensorLike,t = 0.0) -> TensorLike:
        """Return the velocity field u."""
        x = p[..., 0]
        y = p[..., 1]
        v = bm.zeros_like(p)
        R = bm.sqrt(x**2 + y**2) + 1e-12  # Avoid division by zero
        normal_x = x / (R + 1e-12)
        normal_y = y / (R + 1e-12)
        v = bm.stack((normal_x, normal_y), axis=-1)/(- R[:, None])
        return v

    @cartesian
    def init_solution(self, p: TensorLike,t = 0.0) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt((x)**2 + (y)**2)
        r0 = self.r0
        eta = self.eta
        return -bm.tanh((r - r0) / eta)
    