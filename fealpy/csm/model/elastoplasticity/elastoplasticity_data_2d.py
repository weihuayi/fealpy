from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

import numpy as np
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from typing import Tuple

class ElastoplasticityData2D:
    """
    2D Elasto-Plastic Cantilever Beam with von Mises Yield and Ziegler Hardening

    Domain: (0, 1) x (0, 0.25) dm
    Boundary: Clamped at x=0 (Dirichlet), Neumann traction on top surface (y=0.25)
    """

    def __init__(self):
        self.E = 1e5                # Young's modulus in MPa
        self.Ep = 1e5               # Hardening modulus
        self.nu = 0.3               # Poisson's ratio
        self.sigma_y0 = 50.0        # Initial yield stress in MPa
        self.p = 1.7                # Distributed load in N/dm
        self.H = self.Ep           # Effective hardening modulus

        self.dim = 2                # 2D problem
        self.lam = self.compute_lambda()
        self.mu = self.compute_mu()

    def geo_dimension(self):
        return self.dim

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0.0, 1.0), (0.0, 0.25)  # [0, 1] x [0, 0.25] dm

    def compute_lambda(self):
        E, nu = self.E, self.nu
        return E * nu / ((1 + nu) * (1 - 2 * nu))

    def compute_mu(self):
        return self.E / (2 * (1 + self.nu))

    def stress_strain_tensor(self):
        """Return 4th-order elasticity tensor C_{ijkl} for isotropic elasticity."""
        lam, mu = self.lam, self.mu
        def C(eps):
            return lam * bm.trace(eps, axis1=-2, axis2=-1)[..., None, None] * bm.eye(2) + 2 * mu * eps
        return C

    @cartesian
    def body_force(self, x):
        """Return the body force vector f(x)."""
        return bm.zeros((x.shape[0], self.dim))

    @cartesian
    def neumann(self, x):
        """
        Neumann traction on boundary (e.g., top surface y=0.25)
        Uniform vertical load: [0, -p]
        """
        value = bm.zeros((x.shape[0], self.dim))
        value[:, 1] = -self.p  # Apply downward force
        return value

    def dirichlet_boundary(self, x):
        """Clamped boundary at x = 0."""
        return bm.isclose(x[:, 0], 0.0)

    @cartesian
    def dirichlet(self, x):
        """Zero displacement on clamped edge."""
        return bm.zeros((x.shape[0], self.dim))

    def von_mises_yield(self, stress):
        """
        Return von Mises equivalent stress.
        stress: (NE, 2, 2)
        """
        s = stress - bm.trace(stress, axis1=-2, axis2=-1)[..., None, None] / 3 * bm.eye(2)
        s_sq = (s * s).sum(axis=(-2, -1))
        return bm.sqrt(1.5 * s_sq)

    def yield_function(self, stress, alpha):
        """
        Return yield function Φ(σ, α)
        alpha: hardening variable
        """
        return self.von_mises_yield(stress) - (self.sigma_y0 + self.H * alpha)

