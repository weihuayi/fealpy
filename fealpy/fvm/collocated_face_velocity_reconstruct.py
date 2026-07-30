"""Pressure-stabilized face velocity reconstruction for collocated FVM solvers."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .collocated_boundary_conditions import ResolvedRhieChowBoundary
from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import FVMGeometry


class RhieChowInterpolation:
    """Build pressure-stabilized face velocities on collocated grids.

    This operator applies only the standard Rhie-Chow pressure-gradient
    difference to an already reconstructed spatial face velocity.
    """

    def __init__(
        self,
        geometry: FVMGeometry,
        boundary: ResolvedRhieChowBoundary,
    ) -> None:
        if not isinstance(geometry, FVMGeometry):
            raise TypeError("geometry must be an FVMGeometry.")
        if not isinstance(boundary, ResolvedRhieChowBoundary):
            raise TypeError(
                "boundary must be a ResolvedRhieChowBoundary."
            )
        self.geometry = geometry
        self.boundary = boundary

    def pressure_gradient_difference(
        self,
        pressure: TensorLike,
        pressure_gradient: TensorLike,
    ) -> TensorLike:
        """Return the Rhie-Chow pressure-gradient difference.

        This is the difference between the cell-jump pressure gradient along
        the owner-neighbour line and the interpolated reconstructed gradient.
        """
        geometry = self.geometry
        d_f = geometry.d_f
        mag_d_f = geometry.mag_d_f
        face_to_cell = geometry.face_to_cell
        partial_p = (
            pressure[face_to_cell[:, 1]]
            - pressure[face_to_cell[:, 0]]
        ) / mag_d_f
        partial_p = self.boundary.apply_pressure_partial(
            pressure,
            partial_p,
            mag_d_f,
        )
        e_cf = d_f / mag_d_f[:, None]
        overline_grad_p_f = reconstruct_face_gradient(
            geometry,
            pressure_gradient,
            pressure,
            boundary=self.boundary.pressure_state.gradient.boundary,
        )
        interpolated_normal_gradient = bm.einsum("ij,ij->i", overline_grad_p_f, e_cf)
        gradient_difference = (partial_p - interpolated_normal_gradient)[:, None] * e_cf
        return self.boundary.apply_gradient_difference(
            gradient_difference
        )

    def apply(
        self,
        base_face_velocity: TensorLike,
        cell_pressure: TensorLike,
        face_pressure_response: TensorLike,
        cell_pressure_gradient: TensorLike,
    ) -> TensorLike:
        """Apply the pressure-gradient difference to a spatial face field."""
        geometry = self.geometry
        if base_face_velocity.shape != (geometry.NF, geometry.GD):
            raise ValueError(
                "base face velocity must have shape (NF, GD)."
            )
        if cell_pressure.shape != (geometry.NC,):
            raise ValueError("cell pressure must have shape (NC,).")
        if face_pressure_response.shape != (geometry.NF,):
            raise ValueError(
                "face pressure response must have shape (NF,)."
            )
        if cell_pressure_gradient.shape != (geometry.NC, geometry.GD):
            raise ValueError(
                "cell pressure gradient must have shape (NC, GD)."
            )
        gradient_difference = self.pressure_gradient_difference(
            cell_pressure,
            cell_pressure_gradient,
        )
        return (
            base_face_velocity
            - face_pressure_response[:, None] * gradient_difference
        )
