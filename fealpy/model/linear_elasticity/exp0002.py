from typing import Optional
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d



class Exp0002(BoxMesher2d):
    """
    2D Linear Elasticity problem with trigonometric displacement

    -∇·σ = f in Ω
    u = 0 on ∂Ω (homogeneous Dirichlet)

    Displacement:
        u = [sin(πx) sin(πy), 0]^T

    Body force:
        f = [(22.5π²/13) sin(πx) sin(πy), -(12.5π²/13) cos(πx) cos(πy)]^T

    Material:
        E = 1, ν = 0.3
    """

    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
        self.hypo = 'plane_strain'  # 可选：'plane_stress'

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
        u1 = bm.sin(math.pi * x) * bm.sin(math.pi * y)
        u2 = bm.zeros_like(u1)
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        π2 = math.pi ** 2
        f1 = (22.5 * π2 / 13.0) * bm.sin(math.pi * x) * bm.sin(math.pi * y)
        f2 = -(12.5 * π2 / 13.0) * bm.cos(math.pi * x) * bm.cos(math.pi * y)
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
