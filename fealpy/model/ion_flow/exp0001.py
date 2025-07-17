
from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm
from ..mesher.sphere_domain_mesher import SphereDomainMesher

class Exp0001(SphereDomainMesher):
    """
    Ion flow model in a 3D spherical shell domain.
    Implements the fully coupled ion-flow PDE on Ω = {x ∈ ℝ³ : r1 < |x| < r2}.

    PDE system:
        -Δφ - ρ = f1,  Ω = {(x,y,z) | r1 < r < r2}, where r = sqrt(x^2 + y^2 + z^2)
         ∇ρ ⋅ ∇φ - ρ² = f2

    Exact solution:
        φ(x,y,z) = U0 * log(r / r2) / log(r1 / r2)
        ρ(x,y,z) = c1 / sqrt(k3^2 + r^2)
        where r1 = 0.05, r2 = 0.5, U0 = 20, c1 = sqrt(r1 * E_on * eps0 * rho0),
        k2 = sqrt(r1 * E_on * eps0 / rho0), k3 = sqrt(k2^2 - r1^2),
        E_on = 33.7 * m * delta * (1 + 0.24 / sqrt(100 * r1 * delta)) * 100,
        m = 0.4, delta = 1, rho0 = 565.9759, eps0 = 1.0.

    Dirichlet boundary conditions:
        φ(r1) = U0, φ(r2) = 0
        ρ(r1) = c1 / sqrt(k3^2 + r1^2)
        ρ(r2) = c1 / sqrt(k3^2 + r2^2)

    Reference:
        https://wnesm678i4.feishu.cn/wiki/TxdhwkTHiihTdok7dyxc7YdunBg
    """

    def __init__(self, option: dict = {}):
        # Domain parameters
        self.r1 = bm.tensor(option.get('r1', 0.05))
        self.r2 = bm.tensor(option.get('r2', 0.5))
        super().__init__()
        self.U0 = bm.tensor(option.get('U0', 200))
        self.eps0 = bm.tensor(option.get('eps0', 1.0))
        self.rho0 = bm.tensor(option.get('rho0', 565.9759))
        self.m = bm.tensor(option.get('m', 0.4))
        self.delta = bm.tensor(option.get('delta', 1.0))

        self.Eon = 33.7 * self.m * self.delta * (1 + 0.24 * bm.sqrt(100 * self.r1 * self.delta)) * 100
        self.k2 = bm.sqrt(self.r1 * self.Eon * self.eps0 / self.rho0)
        self.k3 = bm.sqrt(self.k2**2 - self.r1**2)
        self.c1 = bm.sqrt(self.r1 * self.Eon * self.eps0 * self.rho0)

    def geo_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # fits the spherical shell inside the unit cube

    def _r(self, p: TensorLike) -> TensorLike:
        # p.shape = (..., 3)
        return bm.linalg.norm(p, axis=-1)

    # === Potential φ and its properties ===
    def potential(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return self.U0 * bm.log(r / self.r2) / bm.log(self.r1 / self.r2)

    def potential_gradient(self, p: TensorLike) -> TensorLike:
        r = self._r(p)[..., None]  # (..., 1)
        coeff = self.U0 / (r * bm.log(self.r1 / self.r2))  # (..., 1)
        return coeff * p  # (..., 3)

    def potential_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.potential(p)

    def is_potential_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return bm.logical_or(bm.isclose(r, self.r1), bm.isclose(r, self.r2))

    # === Density ρ and its properties ===
    def density(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        return self.c1 / bm.sqrt(self.k3**2 + r**2)

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
        term1 = -self.U0 / (r**2 * bm.log(self.r1 / self.r2))
        term2 = -self.c1 / bm.sqrt(self.k3**2 + r**2)
        return term1 + term2

    def source_density(self, p: TensorLike) -> TensorLike:
        r = self._r(p)
        term1 = -self.c1 * self.U0 / ((self.k3**2 + r**2)**1.5 * bm.log(self.r1 / self.r2))
        term2 = -self.c1**2 / (self.k3**2 + r**2)
        return term1 + term2


    