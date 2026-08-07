"""Fixed P0 field layout for collocated incompressible-flow FVM."""

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace, TensorFunctionSpace
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


class CollocatedDiscretization:
    """Own the fixed geometry, P0 spaces, and velocity dof conversion."""

    degree = 0

    def __init__(self, *, geometry: FVMGeometry) -> None:
        if not isinstance(geometry, FVMGeometry):
            raise TypeError("geometry must be an FVMGeometry.")
        self.geometry = geometry
        self.GD = geometry.cell_center.shape[1]
        self.NC = geometry.NC
        self.NF = geometry.NF
        self.space = ScaledMonomialSpace(geometry.mesh, self.degree)
        self.velocity_space = TensorFunctionSpace(
            self.space,
            shape=(self.GD, -1),
        )

    def cell_vector_to_dofs(
        self,
        cell_velocity: TensorLike,
    ) -> TensorLike:
        """Convert physical ``(NC, GD)`` velocity to component-major dofs."""
        if cell_velocity.shape != (self.NC, self.GD):
            raise ValueError("cell velocity must have shape (NC, GD).")
        return bm.swapaxes(cell_velocity, 0, 1).flatten()

    def dofs_to_cell_vector(
        self,
        velocity_dofs: TensorLike,
    ) -> TensorLike:
        """Convert component-major ``(GD*NC,)`` dofs to physical velocity."""
        if velocity_dofs.shape != (self.GD * self.NC,):
            raise ValueError("velocity dofs must have shape (GD*NC,).")
        return bm.stack(
            [
                velocity_dofs[
                    component * self.NC : (component + 1) * self.NC
                ]
                for component in range(self.GD)
            ],
            axis=-1,
        )

    def component_cell_diagonal(
        self,
        diagonal: TensorLike,
    ) -> TensorLike:
        """Repeat one canonical ``(NC,)`` diagonal for every component."""
        if diagonal.shape != (self.NC,):
            raise ValueError("cell diagonal must have shape (NC,).")
        return bm.concatenate([diagonal for _ in range(self.GD)], axis=0)

    def cell_response(
        self,
        cell_diagonal: TensorLike,
    ) -> TensorLike:
        """Return the canonical scalar cell response ``V/a_P``."""
        if cell_diagonal.shape != (self.NC,):
            raise ValueError("cell diagonal must have shape (NC,).")
        return self.geometry.cell_measure / cell_diagonal


__all__ = ["CollocatedDiscretization"]
