from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike
from typing import Tuple
from fealpy.mesh import TriangleMesh

class exp1():
    """
    2D Elasto-Plastic Beam with von Mises Yield and Ziegler Hardening

    Domain: (0, 1) x (0, 0.25) dm
    Left and bottom boundaries are symmetric (clamped),
    Neumann traction applies on the top boundary (y=0.25),
    with time-dependent loading.
    """

    def __init__(self):
        self.E = 206900             # Young's modulus in MPa
        self.nu = 0.29              # Poisson's ratio
        self.a = 10000              # Hardening modulus a in MPa
        self.sigma_y0 = 450 * (2/3)**0.5  # Initial yield stress in MPa
        self.H = self.a             # Ziegler hardening parameter

        self.dim = 2
        self.Ft_max = 2e5           # N: max traction force
        self.n = 2                  # Mesh refine level

        self.lam = self.compute_lambda()
        self.mu = self.compute_mu()

    def compute_lambda(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def compute_mu(self):
        return self.E / (2 * (1 + self.nu))

    def geo_dimension(self):
        return self.dim

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0.0, 1.0), (0.0, 0.25)  # [0,1] x [0,0.25] dm

    def init_mesh(self):
        node = bm.array([
            [-5, -5], [0, -5], [-5, 0],
            [0, 0], [5, 0], [-5, 5],
            [0, 5], [5, 5]
        ], dtype=bm.float64)

        cell = bm.array([
            [1, 3, 0], [2, 0, 3],
            [3, 6, 2], [5, 2, 6],
            [4, 7, 3], [6, 3, 7]
        ], dtype=bm.int32)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(self.n)
        return mesh

    def stress_strain_tensor(self):
        lam, mu = self.lam, self.mu
        def C(eps):
            return lam * bm.trace(eps, axis1=-2, axis2=-1)[..., None, None] * bm.eye(2) + 2 * mu * eps
        return C

    @cartesian
    def body_force(self, x):
        return bm.zeros((x.shape[0], self.dim))  # No body force

    def time_dependent_load(self, t: float) -> float:
        """
        Return the scalar multiplier σ(t) ∈ [0,1] defining time-dependent traction.
        Piecewise linear:
            [1,2] → from 1→0,
            [2,3] → from 0→1,
            [3,4] → from 1→0
        """
        if 1.0 <= t < 2.0:
            return 2.0 - t
        elif 2.0 <= t < 3.0:
            return t - 2.0
        elif 3.0 <= t <= 4.0:
            return 4.0 - t
        else:
            return 0.0

    @cartesian
    def neumann(self, x, t: float = 0.0):
        """
        Time-dependent Neumann boundary condition.
        Vertical traction: [0, -Ft_max * σ(t)]
        """
        value = bm.zeros((x.shape[0], self.dim))
        sigma_t = self.time_dependent_load(t)
        value[:, 1] = -self.Ft_max * sigma_t
        return value

    def dirichlet_boundary(self, x):
        """Clamped boundary: x = 0 or y = 0 (left and bottom edges)"""
        return bm.isclose(x[:, 0], 0.0) | bm.isclose(x[:, 1], 0.0)

    @cartesian
    def dirichlet(self, x):
        return bm.zeros((x.shape[0], self.dim))  # Zero displacement

    def von_mises_yield(self, stress):
        """
        Return von Mises equivalent stress.
        stress: shape (NE, 2, 2)
        """
        s = stress - bm.trace(stress, axis1=-2, axis2=-1)[..., None, None] / 3 * bm.eye(2)
        s_sq = (s * s).sum(axis=(-2, -1))
        return bm.sqrt(1.5 * s_sq)

    def yield_function(self, stress, alpha):
        """
        Yield function: Φ(σ, α) = σ_eq - (σ_y0 + H * α)
        """
        return self.von_mises_yield(stress) - (self.sigma_y0 + self.H * alpha)
