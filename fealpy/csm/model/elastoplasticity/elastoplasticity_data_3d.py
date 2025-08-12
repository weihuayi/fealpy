import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from typing import Tuple

class ElastoplasticityData3D:
    """
    3D Elasto-Plastic Block specimen with pure isotropic hardening.

    Inherits from BoxMesher3d to generate a unit box mesh in 3D, and sets up material and body force parameters
    for a simple linear elasticity eigenvalue example (Exp0001).

    Attributes
        L : float
            Length of the box in the x-direction.
        W : float
            Width of the box in the y- and z-directions.
        g : float
            Scaled gravity acceleration based on aspect ratio.
        d : TensorLike
            Gravity direction vector.
        Geometry: (0,10)x(0,10)x(0,20) mm³
    Material:
        - E = 2.1e5 MPa
        - nu = 0.3
        - Pure isotropic hardening
        - Experimental yield table provided
    """

    def __init__(self):
        # Material parameters
        self.E = 2.1e5       # Young's modulus in MPa
        self.nu = 0.3        # Poisson's ratio
        self.hardening_type = "isotropic"
        self.hardening_ratio = 1.0  # fully isotropic
        self.dim = 3         # 3D problem

        # Experimental yield stress - plastic strain data
        # Columns: [step, yield_stress (MPa), plastic_strain]
        self.yield_table = np.array([
            [1,  50.2, 0.0000],
            [2,  96.0, 0.0235],
            [3, 144.0, 0.0474],
            [4, 224.0, 0.0953],
            [5, 287.0, 0.1377],
            [6, 330.0, 0.1800],
        ])

        # Compute elasticity parameters
        self.lam = self.compute_lambda()
        self.mu = self.compute_mu()

    def geo_dimension(self):
        return 3

    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return geometric domain: 10×10×20 mm³."""
        return (0, 10), (0, 10), (0, 20)

    def compute_lambda(self):
        """Compute Lamé parameter λ."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def compute_mu(self):
        """Compute Lamé parameter μ = shear modulus."""
        return self.E / (2 * (1 + self.nu))

    def elasticity_tensor(self):
        """
        Return the 4-th order elasticity tensor for isotropic elasticity:
        σ = C : ε, with C_{ijkl} = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
        """
        lam, mu = self.lam, self.mu

        def C(eps):
            """eps: (..., 3, 3) strain tensor."""
            trace_eps = bm.trace(eps, axis1=-2, axis2=-1)[..., None, None]
            identity = bm.eye(3)
            return lam * trace_eps * identity + 2 * mu * eps

        return C

    @cartesian
    def body_force(self, x):
        """Default body force: 0."""
        return bm.zeros((x.shape[0], 3))

    @cartesian
    def dirichlet(self, x):
        """Clamped boundary: e.g., z=0 面固定."""
        return bm.zeros((x.shape[0], 3))

    def dirichlet_boundary(self, x):
        """Clamped face at z = 0."""
        return bm.isclose(x[:, 2], 0.0)

    @cartesian
    def neumann(self, x):
        """
        Apply traction on top surface (z=20mm) if needed.
        Default: no Neumann traction.
        """
        return bm.zeros((x.shape[0], 3))

    def von_mises_stress(self, sigma):
        """
        Compute von Mises equivalent stress.
        sigma: (NE, 3, 3) stress tensor
        """
        tr_sigma = bm.trace(sigma, axis1=-2, axis2=-1)[..., None, None]
        dev = sigma - tr_sigma / 3 * bm.eye(3)
        s_sq = (dev * dev).sum(axis=(-2, -1))
        return bm.sqrt(1.5 * s_sq)

    def hardening_function(self, plastic_strain):
        """
        Given equivalent plastic strain, interpolate yield stress.
        """
        plastic_strain_table = self.yield_table[:, 2]
        yield_stress_table = self.yield_table[:, 1]
        return np.interp(plastic_strain, plastic_strain_table, yield_stress_table)

    def yield_function(self, sigma, alpha):
        """
        Yield function: φ(σ, α) = σ_vm - σ_y(α)
        """
        sigma_vm = self.von_mises_stress(sigma)
        sigma_y = self.hardening_function(alpha)
        return sigma_vm - sigma_y
