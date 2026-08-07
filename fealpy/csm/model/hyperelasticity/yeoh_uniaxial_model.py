from typing import Optional, Dict, Any, Tuple

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher3d


class YeohUniaxialModel(BoxMesher3d):
    """One-element uniaxial deformation model for Yeoh hyperelastic benchmark.
    
    This model is designed for the CSM benchmark case: RD-V: 0210 Yeoh Hyperelastic Material.
    """
    
    def __init__(
        self,
        length: float = 1.0,
        stretch: float = 1.0,
        axis: int = 0,
        tol: float = 1.0e-12
    ):
        """Initialize the uniaxial Yeoh benchmark model.

        Parameters:
            length(float): Edge length of the cubic domain. Default is 1.0.
            stretch(float): Prescribed stretch ratio along the loading direction.
                - stretch = current_length / initial_length.
                - stretch > 1 means tension, stretch < 1 means compression.
            axis(int): Loading direction. 0 for x, 1 for y, 2 for z.
            tol(float): Geometric tolerance for boundary detection.
        """
        
        self.length = length
        self.stretch = stretch
        self.axis = axis
        self.tol = tol
        
        super().__init__(box=[0.0, self.length, 0.0, self.length, 0.0, self.length])
    
    def __str__(self) -> str:
        sx, sy, sz = self.principal_stretches()
        return (
            f"\n  Yeoh uniaxial hyperelastic benchmark model:\n"
            f"  Domain: [0, {self.length}] x [0, {self.length}] x [0, {self.length}]\n"
            f"  Loading axis: {self.axis}\n"
            f"  Prescribed stretch: {self.stretch}\n"
            f"  Principal stretches: ({sx}, {sy}, {sz})\n"
            f"  Assumption: incompressible uniaxial deformation."
        )
        
    def geo_dimension(self) -> int:
        """Return geometric dimension."""
        return 3
    
    def principal_stretches(self):
        """Return the three principal stretches for incompressible uniaxial deformation.
        
        For the Yeoh uniaxial benchmark, the loading-direction stretch is lambda,
        and the two lateral stretches are lambda^(-1/2), so that:
                J = lambda_1 * lambda_2 * lambda_3 = 1
        
        Returns:
            Tuple[float, float, float]: Principal stretches in x, y and z directions.
        """
        lam = self.stretch
        
        if lam <= 0:
            raise ValueError("stretch must be positive.")

        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2.")

        lam_t = lam ** (-0.5)

        stretches = [lam_t, lam_t, lam_t]
        stretches[self.axis] = lam

        return stretches[0], stretches[1], stretches[2]
    
    def deformation_gradient(self) -> TensorLike:
        """Return the homogeneous deformation gradient F."""
        sx, sy, sz = self.principal_stretches()
        
        F = bm.array(
            [[sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, sz],
            ],dtype=bm.float64)

        return F
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        """Body force term of the PDE."""
        return bm.zeros_like(p, **bm.context(p))
    
    @cartesian
    def displacement(self, p: TensorLike) -> TensorLike:
        """Exact displacement field for homogeneous uniaxial deformation.

        The deformation map is:
            x = F X
        so the displacement is:
            u(X) = (F - I) X

        For incompressible uniaxial deformation:

            lambda_loading = stretch
            lambda_lateral = stretch ** (-1/2)
        """
        sx, sy, sz = self.principal_stretches()

        ux = (sx - 1.0) * p[..., 0]
        uy = (sy - 1.0) * p[..., 1]
        uz = (sz - 1.0) * p[..., 2]

        return bm.stack([ux, uy, uz], axis=-1)
    
    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        """Dirichlet displacement boundary condition.

        For this Yeoh one-element benchmark, the deformation is prescribed
        through displacement boundary conditions.
        """
        return self.displacement(p)
    
    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        """Return True on the displacement boundary.

        In the homogeneous deformation benchmark, all boundary points are
        prescribed by the analytical displacement field.
        """
        return self.is_boundary(p)

    @cartesian
    def is_boundary(self, p: TensorLike) -> TensorLike:
        pass