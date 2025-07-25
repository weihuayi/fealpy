from typing import Optional

from ....backend import bm
from ....typing import TensorLike
from ....decorator import cartesian
from ....material import LinearElasticMaterial
from ....mesher import BoxMesher3d


class Exp0001(BoxMesher3d):
    """Example class defining a 3D box mesh and physical parameters for test case Exp0001.

    This class inherits from BoxMesher3d to generate a unit box mesh in 3D,
    and sets up material and body force parameters for a simple linear elasticity
    eigenvalue example (Exp0001).

    Attributes
        L : float
            Length of the box in the x-direction.
        W : float
            Width of the box in the y- and z-directions.
        g : float
            Scaled gravity acceleration based on aspect ratio.
        d : TensorLike
            Gravity direction vector.
    """

    def __init__(self, options = {'L':1.0, 'W':0.2}):
        self.L = options['L']
        self.W = options['W']
        super().__init__(box=[0, L, 0, W , 0, W])

        delta = self.W / self.L  # aspect ratio
        self.g = 0.4 * delta**2  # gravity acceleration
        self.d = bm.array(
            [0.0, 0.0, 1.0],
            dtype=bm.float64
        )  # TODO: specify computation device

        # Initialize the material with constant parameters
        self.material = LinearElasticMaterial("box", 
                                              lame_lambda=1.25,
                                              shear_modulus=1.0, 
                                              density=1.0)

    def __str__(self) -> str:
        """Return a multi-line summary including PDE type and key params."""
        # make direction printable
        direction = self.d.tolist() if hasattr(self.d, "tolist") else self.d
        return (
            f"\nExp0001 (3D Linear Elasticity PDE on box domain)\n"
            f"  Box dimensions: L = {self.L}, W = {self.W}\n"
            f"  Gravity: g = {self.g:.3g}, d = {direction}"
        )

    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        """Compute the body force vector at given points.

        For Exp0001, the body force is zero everywhere.

        Parameters
            p : TensorLike
                Coordinates at which to evaluate the body force.

        Returns
            force : TensorLike
                Zero tensor of same shape as p.
        """
        val = bm.zeros_like(p, **bm.context(p))
        return val

    @cartesian
    def displacement(self, p: TensorLike) -> TensorLike:
        """Analytical displacement field (not implemented).

        This method should return the analytical displacement at points p.

        Parameters
            p : TensorLike
                Coordinates at which to evaluate the displacement.

        Raises
            NotImplementedError
                Always, as displacement is not implemented for Exp0001.
        """
        raise NotImplementedError(
            "Displacement computation is not implemented for Exp0001."
        )

    @cartesian
    def strain(self, p: TensorLike) -> TensorLike:
        """Analytical strain tensor (not implemented).

        This method should return the strain tensor at points p.

        Parameters
            p : TensorLike
                Coordinates at which to evaluate the strain.

        Raises
            NotImplementedError
                Always, as strain is not implemented for Exp0001.
        """
        raise NotImplementedError(
            "Strain computation is not implemented for Exp0001."
        )

    @cartesian
    def stress(self, p: TensorLike) -> TensorLike:
        """Analytical stress tensor (not implemented).

        This method should return the stress tensor at points p.

        Parameters
            p : TensorLike
                Coordinates at which to evaluate the stress.

        Raises
            NotImplementedError
                Always, as stress is not implemented for Exp0001.
        """
        raise NotImplementedError(
            "Stress computation is not implemented for Exp0001."
        )

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition for displacement.

        Returns zero displacement on the boundary.

        Parameters
            p : TensorLike
                Coordinates on the boundary.

        Returns
            bc_val : TensorLike
                Zero displacement same shape as p.
        """
        return bm.zeros_like(p, **bm.context(p))

    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        """Indicator function for displacement boundary.

        Identifies points where x == 0 within a tolerance as Dirichlet boundary.

        Parameters
            p : TensorLike
                Coordinates to test.

        Returns
            mask : TensorLike
                Boolean mask where True indicates boundary points.
        """
        return bm.abs(p[..., 0]) < 1e-12

