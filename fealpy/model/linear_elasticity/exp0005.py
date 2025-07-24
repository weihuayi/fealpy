from typing import Optional
from ...backend import backend_manager as bm
from ...decorator import cartesian, variantmethod
from ...typing import  TensorLike
from ...mesher import BoxMesher3d

class Exp0005(BoxMesher3d):
    """
    3D Linear Elasticity problem

    -∇·σ = f  in Ω
    where σ = λ(∇·u)I + 2με is the stress tensor, ε is the strain tensor.

    with the displacement field is:
    u_x = 1e-3 * (2x + y + z) / 2
    u_y = 1e-3 * (x + 2y + z) / 2
    u_z = 1e-3 * (x + y + 2z) / 2

    Material properties:
    - λ = 4e5 MPa (First Lamé parameter)
    - μ = 4e5 MPa (Shear modulus)

    Boundary conditions:
    - Non-homogeneous Dirichlet boundary conditions on all boundaries ∂Ω
    - u = u_exact on ∂Ω (prescribed displacement from the exact solution)
    - No traction (Neumann) boundary conditions
    """

    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        super().__init__(self.box)
        self.hypo = '3D'  # Hypothesis for the problem

    def geo_dimension(self):
        return 3

    @property
    def lam(self, p: Optional[TensorLike] = None) -> TensorLike:
        """First Lamé parameter λ = 4e5 MPa."""
        return 4e5
    @property
    def mu(self, p: Optional[TensorLike] = None) -> TensorLike:
        """Second Lamé parameter μ (shear modulus) = 4e5 MPa."""
        return 4e5
    @property
    def rho(self, p: Optional[TensorLike] = None) -> TensorLike:
        """Material density ρ = 1e4 kg/m³."""
        return 1e4

    def rho(self, p: Optional[TensorLike] = None) -> TensorLike:
        """Material density ρ = 1e4 kg/m³."""
        return 1e4

    @cartesian
    def body_force(self, p: TensorLike):
        val = bm.zeros(p.shape, dtype=p.dtype, device=bm.get_device(p))
        return val

    @cartesian
    def displacement(self, p: TensorLike):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        u_x = 1e-3 * (2*x + y + z) / 2
        u_y = 1e-3 * (x + 2*y + z) / 2
        u_z = 1e-3 * (x + y + 2*z) / 2
        
        return bm.stack([u_x, u_y, u_z], axis=-1)

    @cartesian
    def strain(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1]
        diag = bm.full(
                    shape, 1e-3, 
                    dtype=p.dtype, device=bm.get_device(p)
                )
        off_diag = bm.full(
                        shape, 1e-3/2, 
                        dtype=p.dtype, device=bm.get_device(p)
                    )

        elements = bm.stack([diag, off_diag, off_diag,
                            off_diag, diag, off_diag,
                            off_diag, off_diag, diag], axis=-1)
        
        return bm.reshape(elements, shape + (3, 3))

    @cartesian
    def stress(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1]

        lam = self.lam(p)
        mu = self.mu(p) 
        
        tr_eps = 3e-3

        diag = bm.full(
                    shape, lam * tr_eps + 2 * mu * 1e-3,
                    dtype=p.dtype, device=bm.get_device(p)
                )
        off_diag = bm.full(
                        shape, 2 * mu * 1e-3 / 2,
                        dtype=p.dtype, device=bm.get_device(p)
                    )
        elements = bm.stack([diag, off_diag, off_diag,
                            off_diag, diag, off_diag,
                            off_diag, off_diag, diag], axis=-1)
        
        return bm.reshape(elements, shape + (3, 3))

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        result = self.displacement(p)
        return result
    
    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        eps = 1e-12
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        
        xmin, xmax, ymin, ymax, zmin, zmax = self.box
        
        on_x_min = bm.abs(x - xmin) < eps
        on_x_max = bm.abs(x - xmax) < eps
        on_y_min = bm.abs(y - ymin) < eps
        on_y_max = bm.abs(y - ymax) < eps
        on_z_min = bm.abs(z - zmin) < eps
        on_z_max = bm.abs(z - zmax) < eps
        
        return on_x_min | on_x_max | on_y_min | on_y_max | on_z_min | on_z_max
    
    @cartesian
    def traction_bc(self, p: TensorLike) -> TensorLike:
        pass

    @cartesian
    def is_traction_boundary(self, p: TensorLike) -> TensorLike:
        pass
