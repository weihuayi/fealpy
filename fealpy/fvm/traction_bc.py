"""Finite-volume steady momentum traction boundary algebra."""

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class TractionBC:
    """Apply prescribed steady-momentum traction to FVM boundary algebra.

    The class consumes already-defined PDE boundary data.  Engineering patch
    mapping, SIMPLE pressure-correction closure, and Rhie--Chow interpolation
    rules remain outside this boundary operator.
    """

    geometry: FVMGeometry
    faces: TensorLike
    face_average_values: TensorLike
    cell_source: TensorLike

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FVMGeometry):
            raise TypeError("geometry must be an FVMGeometry.")
        if self.faces.ndim != 1:
            raise ValueError("faces must have shape (N,).")
        expected = (self.faces.shape[0], self.geometry.GD)
        if self.face_average_values.shape != expected:
            raise ValueError(
                f"face_average_values must have shape {expected}."
            )
        if self.cell_source.shape != (self.geometry.NC, self.geometry.GD):
            raise ValueError("cell_source must have shape (NC, GD).")

    def source(self, reference: TensorLike) -> TensorLike:
        """Return the cell source in the caller's tensor context."""
        return bm.array(
            self.cell_source,
            dtype=reference.dtype,
            device=bm.get_device(reference),
        )

    def apply_pressure_force(
        self,
        cell_force,
        pressure,
        pressure_gradient,
    ):
        """Remove the reconstructed pressure face force on traction faces."""
        faces = self.faces
        owner = self.geometry.owner[faces]
        displacement = (
            self.geometry.face_center[faces]
            - self.geometry.cell_center[owner]
        )
        pressure_trace = pressure[owner] + bm.einsum(
            "ij,ij->i",
            pressure_gradient[owner],
            displacement,
        )
        return bm.index_add(
            cell_force,
            owner,
            pressure_trace[:, None] * self.geometry.S_f[faces],
            axis=0,
            alpha=-1,
        )

    def convection_source(
        self,
        convection_face_velocity,
        cell_velocity,
        reconstructed_face_velocity,
    ):
        """Return the deferred traction-face convection correction."""
        faces = self.faces
        owner = self.geometry.owner[faces]
        flux = bm.einsum(
            "ij,ij->i",
            convection_face_velocity[faces],
            self.geometry.S_f[faces],
        )
        correction = (
            reconstructed_face_velocity[faces]
            - cell_velocity[owner]
        )
        cell_source = bm.zeros_like(cell_velocity)
        return bm.index_add(
            cell_source,
            owner,
            -flux[:, None] * correction,
            axis=0,
        )

    def apply_cross_diffusion_flux(self, face_flux):
        """Suppress explicit cross-diffusion flux on traction faces."""
        faces = self.faces
        return bm.set_at(
            face_flux,
            faces,
            bm.zeros_like(face_flux[faces]),
        )
