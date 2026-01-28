
from typing import Tuple, Callable

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

from fealpy.mesher import LshapeMesher

class ThickWalledCylinderData(LshapeMesher):
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

    def __init__(self, center=(0.0, 0.0), inner_radius=100.0):
        self.E = 21000         # Young's modulus in MPa
        self.nu = 0.3            # Poisson's ratio
        self.hardening_modulus = 0              # Hardening modulus a in MPa
        self.yield_stress = 32 # Initial yield stress in MPa
        self.center = center
        self.inner_radius = inner_radius
        self.pressure = 15.0     # Applied internal pressure in MPa

        self.dim = 2
        self.Ft_max = 15        # N: max traction force
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
    def neumann_boundary(self, p):
        cx, cy = self.center[0], self.center[1]
        dx = p[..., 0] - cx
        dy = p[..., 1] - cy
        r = bm.sqrt(dx**2 + dy**2)
        return bm.abs(r - self.inner_radius) < 0.1

    @cartesian
    def neumann(self, p):
        cx, cy = self.center[0], self.center[1]
        dx = p[..., 0] - cx
        dy = p[..., 1] - cy
        r = bm.sqrt(dx**2 + dy**2)
        nx = dx / r
        ny = dy / r
        # 构造 (..., 2) 形状的张量
        pressure = bm.array(self.pressure, dtype=p.dtype)
        traction = bm.stack([nx, ny], axis=-1)
        result = pressure * traction
        return bm.array(result, dtype=p.dtype)
    
    """
    
    @cartesian
    def dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (bm.abs(x-0) < 1e-8) | (bm.abs(y-0) < 1e-8)
        return flag
    
    @cartesian
    def dirichlet(self, p):
        return bm.zeros_like(p)

    """
    @cartesian
    def dirichlet_boundary(self, p):
        """
        Apply symmetry boundary conditions on x=0 and y=0 planes.
        Returns a boolean mask for points on Dirichlet boundaries.
        """
        x = p[..., 0]
        y = p[..., 1]
        # Points on x=0 (y-axis) OR y=0 (x-axis)
        return (bm.abs(x) < 1e-8) | (bm.abs(y) < 1e-8)

    @cartesian
    def dirichlet(self, p):
        """
        Symmetry condition:
          - On x=0: u_x = 0, u_y free
          - On y=0: u_y = 0, u_x free
        We return [u_x, u_y] with appropriate zeros.
        """
        x = p[..., 0]
        y = p[..., 1]
        
        # Initialize displacement as zero
        ux = bm.zeros_like(x)
        uy = bm.zeros_like(y)
        
        # Enforce ux = 0 on x=0
        ux = bm.where(bm.abs(x) < 1e-8, 0.0, ux)
        # Enforce uy = 0 on y=0
        uy = bm.where(bm.abs(y) < 1e-8, 0.0, uy)

        return bm.stack([ux, uy], axis=-1)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:

        x = points[..., 0]

        coord = bm.abs(x) < self._eps

        return coord

    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:

        y = points[..., 1]

        coord = bm.abs(y) < self._eps

        return coord

    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable]:
        """左侧对称约束 (u_x = 0), 右下角滑移支座 (u_y = 0)"""
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)

        return bm.zeros(points.shape, **kwargs)
   