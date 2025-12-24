
from typing import Tuple, Callable

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike


class PerforatedSquarePlateData():
    """
    2D Elasto-Plastic Beam with von Mises Yield and Ziegler Hardening

    Attributes:
        E : float
            Young's modulus (MPa).
            Hardening modulus (MPa).
            Initial yield stress (MPa).
        H : float
            Ziegler hardening parameter.
        dim : int
            Spatial dimension (2 for 2D).
        Ft_max : float
            Maximum traction force (N).
        n : int
            Mesh refinement level.

    Domain: 
    The left and bottom boundaries are clamped (zero displacement).
    A time-dependent Neumann traction is applied on the top boundary (y=0.25).
    """

    def __init__(self):
        self.E = 100000         # Young's modulus in MPa
        self.nu = 0.25           # Poisson's ratio
        self.hardening_modulus = 0              # Hardening modulus a in MPa
        self.yield_stress = 1100 # Initial yield stress in MPa
        self.pressure = 500     # Applied internal pressure in MPa

        self.dim = 3
        self.Ft_max = 500        # N: max traction force
        self._eps = 1e-8

        self.lam = self.compute_lambda()
        self.mu = self.compute_mu()
        
    def __str__(self) -> str:
        """Return a multi-line summary including PDE type and key params."""
        return (
            f"\n  elastoplasticity (2D Elasto-Plastic)\n"
            f"  Box dimensions: L = 5.0, W = 5.0 m\n"
            f"  Young's modulus: E = {self.E} MPa\n"
            f"  Poisson's ratio: nu = {self.nu}\n"
            f"  Hardening modulus: a = {self.a} MPa\n"
            f"  Initial yield stress: sigma_y0 = {self.sigma_y0} MPa\n"
            f"  Ziegler hardening parameter: H = {self.H}\n"
        )

    def geo_dimension(self):
        return self.dim

    def compute_lambda(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def compute_mu(self):
        return self.E / (2 * (1 + self.nu))

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.box  
         
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1]
        f1 = bm.zeros(shape)       
        f2 = bm.zeros(shape)
        return bm.stack([f1, f2], axis=-1)
    


    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        对称边界：x=0 平面 (ux=0), y=0 平面 (uy=0)
        p: (..., 3) 的点坐标
        返回布尔掩码，标记所有 Dirichlet 约束点（用于整体位移为零的旧接口，可保留但不用于分量约束）
        """
        x = p[..., 0]
        y = p[..., 1]
        return (bm.abs(x) < self._eps) | (bm.abs(y) < self._eps)

    @cartesian
    def dirichlet_bc(self, p):
        """
        【注意】在分量约束下，此函数通常不被直接用于求解，
        而是由 is_dirichlet_boundary_dof_* 控制。
        但为兼容性，仍返回全零位移。
        """
        return bm.zeros_like(p)  # shape (..., 3)

    # ========================
    # 分量级 Dirichlet 约束（关键！）
    # ========================

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """在 x=0 平面上固定 ux"""
        x = points[..., 0]
        return bm.abs(x) < self._eps

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        """在 y=0 平面上固定 uy"""
        y = points[..., 1]
        return bm.abs(y) < self._eps

    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        z = points[..., 2]
        return bm.abs(z) < self._eps  # 假设 z=0 是底面

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable, Callable]:
        """返回三个方向的 Dirichlet 判断函数"""
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
            self.is_dirichlet_boundary_dof_z
        )

    # ========================
    # Neumann 边界（均布载荷）
    # ========================

    @cartesian
    def neumann_boundary(self, p):
        """
        在 x = 5.0 的右侧面施加 Neumann 载荷
        """
        x = p[..., 0]
        Lx_half = 10.0  
        return bm.abs(x - Lx_half) < self._eps

    @cartesian
    def neumann(self, p):
        """
        施加面力 t = (p, 0, 0)，其中 p = 1000 N/m²
        """
        pressure = 1000.0  # N/m²
        # 构造 (..., 3) 的牵引力向量
        traction = bm.stack([
            bm.full(p.shape[:-1], pressure, dtype=p.dtype),  # tx = p
            bm.zeros(p.shape[:-1], dtype=p.dtype),           # ty = 0
            bm.zeros(p.shape[:-1], dtype=p.dtype)            # tz = 0
        ], axis=-1)
        return traction