
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

from fealpy.mesh import TriangleMesh

class ElastoplasticityData2D:
    """
    2D Elasto-Plastic Beam with von Mises Yield and Ziegler Hardening

    Attributes:
        E : float
            Young's modulus (MPa).
        nu : float
            Poisson's ratio.
        a : float
            Hardening modulus (MPa).
        sigma_y0 : float
            Initial yield stress (MPa).
        H : float
            Ziegler hardening parameter.
        dim : int
            Spatial dimension (2 for 2D).
        Ft_max : float
            Maximum traction force (N).
        n : int
            Mesh refinement level.

    Domain: (0, 1) x (0, 0.25) dm.
    The left and bottom boundaries are clamped (zero displacement).
    A time-dependent Neumann traction is applied on the top boundary (y=0.25).
    """

    def __init__(self):
        self.E = 206900             # Young's modulus in MPa
        self.nu = 0.29              # Poisson's ratio
        self.hardening_modulus = 10000              # Hardening modulus a in MPa
        self.yield_stress = 450 * (2/3)**0.5  # Initial yield stress in MPa

        self.dim = 2
        self.Ft_max = 2e5           # N: max traction force
        self.n = 1                  # Mesh refine level

        self.lam = self.compute_lambda()
        self.mu = self.compute_mu()
        
    def __str__(self) -> str:
        """Return a multi-line summary including PDE type and key params."""
        return (
            f"\n  elastoplasticity (2D Elasto-Plastic)\n"
            f"  Box dimensions: L = 5.0, W = 5.0 dm\n"
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

    @cartesian
    def body_force(self, x):
        return bm.zeros((x.shape[0], self.dim))  # No body force
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1]
        f1 = bm.zeros(shape)       
        f2 = self.Ft_max * bm.ones(shape) 
        return bm.stack([f1, f2], axis=-1)

    @cartesian
    def source_term(self, t: float) -> float:
        """
        Time-dependent source term for the body force.
        This is a placeholder and can be modified as needed.
        
        Parameters:
            t (float): Time variable.   
            
        Returns:
            float: Source term value.
        """
        if 0.0 <= t < 1.0:
            return 200 * t
        elif 1.0 <= t < 2.0:
            return 200 * (2 - t)
        elif 2.0 <= t < 3.0:
            return 200 * (2.0 - t)
        elif 3.0 <= t <= 4.0:
            return 200 * (t - 4.0)
        else:
            return 0.0
        
    @cartesian
    def neumann_boundary(self, x):
        y = x[:, 1]
        return bm.abs(y - 5.0) < 1e-12

    @cartesian
    def neumann(self, p):
        """
        Apply traction force on the top boundary y = 5, unit: MPa.
        """
        shape = p.shape[:-1]
        f1 = bm.zeros(shape)  # x 方向无牵引
        f2 = self.Ft_max * bm.ones(shape)/10.0  # y 方向
        return bm.stack([f1, f2], axis=-1)

    def dirichlet_boundary(self, x):
        """
        Check if points are on the Dirichlet boundary (left and bottom edges).
        
        Parameters:
            x (TensorLike): Points in the domain.
            
        Returns:
            TensorLike: Boolean array indicating if points are on the Dirichlet boundary.
        """
        return bm.abs(x[:, 0] + 5) < 1e-12 | bm.abs(x[:, 1] + 5) < 1e-12
    
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
