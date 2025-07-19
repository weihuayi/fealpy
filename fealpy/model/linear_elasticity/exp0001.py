from typing import Optional
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d


class Exp0001(BoxMesher2d):
    """
    2D Linear Elasticity problem with polynomial displacement

    -∇·σ = f in Ω
    u = 0 on ∂Ω (homogeneous Dirichlet)

    Material parameters:
    E = 1, ν = 0.3
    """

    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
        self.hypo = 'plane_strain'  # Hypothesis for the problem

    def geo_dimension(self):
        return 2

    def E(self, p: Optional[TensorLike] = None) -> float:
        return 1.0

    def nu(self, p: Optional[TensorLike] = None) -> float:
        return 0.3

    def lam(self, p: Optional[TensorLike] = None) -> float:
        """λ = Eν / ((1+ν)(1-2ν))"""
        E = self.E()
        nu = self.nu()
        return E * nu / ((1 + nu) * (1 - 2 * nu))

    def mu(self, p: Optional[TensorLike] = None) -> float:
        """μ = E / (2(1+ν))"""
        E = self.E()
        nu = self.nu()
        return E / (2 * (1 + nu))

    @cartesian
    def displacement(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u1 = x * (1 - x) * y * (1 - y)
        u2 = bm.zeros_like(u1)
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        f1 = (35 / 13) * y - (35 / 13) * y**2 + (10 / 13) * x - (10 / 13) * x**2
        f2 = - (25 / 26) * (1 - 2 * x) * (1 - 2 * y)
        return bm.stack([f1, f2], axis=-1)

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        return self.displacement(p)

    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        eps = 1e-12
        x, y = p[..., 0], p[..., 1]
        xmin, xmax, ymin, ymax = self.box
        on_x = (bm.abs(x - xmin) < eps) | (bm.abs(x - xmax) < eps)
        on_y = (bm.abs(y - ymin) < eps) | (bm.abs(y - ymax) < eps)
        return on_x | on_y
