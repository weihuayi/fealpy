
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

from fealpy.mesher import LshapeMesher

class ElastoplasticityData2D(LshapeMesher):
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
        self.E = 206900             # Young's modulus in MPa
        self.nu = 0.29              # Poisson's ratio
        self.hardening_modulus = 10000              # Hardening modulus a in MPa
        self.yield_stress = 450 # Initial yield stress in MPa

        self.dim = 2
        self.Ft_max = 200        # N: max traction force

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
        y = p[:, 1]
        return bm.abs(y - 5.0) < 1e-12

    @cartesian
    def neumann(self, p):
        """
        Apply traction force on the top boundary y = 5, unit: MPa.
        """
        shape = p.shape[:-1]
        f1 = bm.zeros(shape)  # x 方向无牵引
        f2 = self.Ft_max * bm.ones(shape)  # y 方向
        return bm.stack([f1, f2], axis=-1)

    def dirichlet_boundary(self, p):
        """
        Check if points are on the Dirichlet boundary (left and bottom edges).
        
        Parameters:
            p (TensorLike): Points in the domain.

        Returns:
            TensorLike: Boolean array indicating if points are on the Dirichlet boundary.
        """
        return (bm.abs(p[:, 0] + 5) < 1e-12) | (bm.abs(p[:, 1] + 5) < 1e-12)

    @cartesian
    def dirichlet(self, p):
        """
        Dirichlet boundary condition: fixed boundary condition (zero displacement).
        This function returns zero displacement for all points on the Dirichlet boundary.
        
        Parameters:
            p (TensorLike): Points in the domain.
            
        Returns:
            TensorLike: Zero displacement for the Dirichlet boundary.
        """
        return bm.zeros_like(p)  # 固定边界条件为零位移
