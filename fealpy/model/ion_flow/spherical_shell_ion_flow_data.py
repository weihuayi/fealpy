
from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm


class SphericalShellIonFlowData:
    """
    Ion flow model in a 3D spherical shell domain.
    Implements the fully coupled ion-flow PDE on Ω = {x ∈ ℝ³ : r1 < |x| < r2}.
    """

    def __init__(self):
        # Domain parameters
        self.r1 = 0.05
        self.r2 = 0.5
        self.U0 = 200
        self.eps0 = 1.0
        self.rho0 = 565.9759
        self.m = 0.4
        self.delta = 1.0

        self.Eon = 33.7 * self.m * self.delta * (1 + 0.24 * np.sqrt(100 * self.r1 * self.delta)) * 100
        self.k2 = np.sqrt(self.r1 * self.Eon * self.eps0 / self.rho0)
        self.k3 = np.sqrt(self.k2**2 - self.r1**2)
        self.c1 = np.sqrt(self.r1 * self.Eon * self.eps0 * self.rho0)

    def geo_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # fits the spherical shell inside the unit cube

    def _r(self, p: TensorLike) -> TensorLike:
        # p.shape = (..., 3)
        return np.linalg.norm(p, axis=-1)

    # === Potential φ and its properties ===
    def potential(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return self.U0 * np.log(r / self.r2) / np.log(self.r1 / self.r2)

    def potential_gradient(self, p: TensorLike) -> TensorLike:
        r = self._r(p)[..., None]  # (..., 1)
        coeff = self.U0 / (r * np.log(self.r1 / self.r2))  # (..., 1)
        return coeff * p  # (..., 3)

    def potential_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.potential(p)

    def is_potential_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return np.logical_or(np.isclose(r, self.r1), np.isclose(r, self.r2))

    # === Density ρ and its properties ===
    def density(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return self.c1 / np.sqrt(self.k3**2 + r**2)

    def density_gradient(self, p: TensorLike) -> TensorLike:
        r = self._r(p)  # (...,)
        denom = (self.k3**2 + r**2)**1.5  # (...,)
        coeff = -self.c1 * r / denom  # (...,)
        coeff = coeff[..., None] / r[..., None]  # (...,1)
        return coeff * p  # (..., 3)

    def density_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.density(p)

    def is_density_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_potential_dirichlet_boundary(p)

    # === RHS Source Terms f1, f2 ===
    def source_potential(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        term1 = -self.U0 / (r**2 * np.log(self.r1 / self.r2))
        term2 = -self.c1 / np.sqrt(self.k3**2 + r**2)
        return term1 + term2

    def source_density(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        term1 = -self.c1 * self.U0 / ((self.k3**2 + r**2)**1.5 * np.log(self.r1 / self.r2))
        term2 = -self.c1**2 / (self.k3**2 + r**2)
        return term1 + term2

