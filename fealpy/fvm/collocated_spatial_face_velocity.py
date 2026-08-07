"""Spatial face-velocity reconstruction for collocated FVM solvers."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .collocated_boundary_conditions import ResolvedVelocityBoundary
from .collocated_discretization import CollocatedDiscretization
from .face_flux_reconstruct import FaceFluxReconstruct
from .fvm_geometry import (
    FVMGeometry,
    face_interpolation_owner_weight,
    interpolate_cell_to_face,
)


def reconstruct_second_order_face_velocity(
    geometry: FVMGeometry,
    gradient: TensorLike,
    cell_velocity: TensorLike,
) -> TensorLike:
    """Blend owner and neighbour affine reconstructions at face centres."""
    owner = geometry.owner
    neighbour = geometry.neighbour
    owner_delta = geometry.face_center - geometry.cell_center[owner]
    neighbour_delta = geometry.face_center - geometry.cell_center[neighbour]
    owner_value = cell_velocity[owner] + bm.einsum(
        "ncd,nd->nc",
        gradient[owner],
        owner_delta,
    )
    neighbour_value = cell_velocity[neighbour] + bm.einsum(
        "ncd,nd->nc",
        gradient[neighbour],
        neighbour_delta,
    )
    owner_weight = face_interpolation_owner_weight(
        geometry,
        method="linear",
    )
    return (
        owner_weight[:, None] * owner_value
        + (1.0 - owner_weight[:, None]) * neighbour_value
    )


class CollocatedSpatialFaceVelocity:
    """Reconstruct face velocity before pressure stabilization.

    The operator owns the immutable spatial scheme, the velocity-gradient
    reconstruction, the optional conservative face-flux correction, and the
    resolved velocity boundary values.  Pressure and pressure response are
    deliberately outside this object.
    """

    def __init__(
        self,
        *,
        discretization: CollocatedDiscretization,
        boundary: ResolvedVelocityBoundary,
        scheme: str,
        interpolation: str,
        face_flux_reconstruct: FaceFluxReconstruct,
    ) -> None:
        self.discretization = discretization
        self.boundary = boundary
        self.scheme = scheme
        self.interpolation = interpolation
        self.face_flux_reconstruct = face_flux_reconstruct

    def interpolate(self, cell_velocity: TensorLike) -> TensorLike:
        """Interpolate canonical cell velocity to every face."""
        return interpolate_cell_to_face(
            cell_velocity,
            geometry=self.discretization.geometry,
            method=self.interpolation,
        )

    def second_order_reconstruct(
        self,
        cell_velocity: TensorLike,
    ) -> TensorLike:
        """Blend owner and neighbour affine reconstructions at face centres."""
        gradient = self.boundary.gradient.cell_gradient(cell_velocity)
        return reconstruct_second_order_face_velocity(
            self.discretization.geometry,
            gradient,
            cell_velocity,
        )

    def reconstruct(self, cell_velocity: TensorLike) -> TensorLike:
        """Return the configured spatial face velocity with resolved boundary."""
        if self.scheme == "interpolated":
            face_velocity = self.interpolate(cell_velocity)
        else:
            face_velocity = self.second_order_reconstruct(cell_velocity)

        if self.face_flux_reconstruct.method != "none":
            flux_defect = self.face_flux_reconstruct.correction(
                cell_velocity,
                face_velocity,
                boundary_face_average=(
                    self.boundary.dirichlet_face_values
                ),
                boundary_faces=self.boundary.dirichlet_operator.faces,
            )
            current_flux = self.compute_flux(face_velocity)
            face_velocity = self.enforce_flux(
                face_velocity,
                current_flux,
                current_flux + flux_defect,
            )
        return self.enforce_boundary(face_velocity)

    def enforce_boundary(
        self,
        face_velocity: TensorLike,
    ) -> TensorLike:
        """Set the already-resolved velocity trace on Dirichlet faces."""
        return bm.set_at(
            face_velocity,
            self.boundary.dirichlet_operator.faces,
            self.boundary.dirichlet_face_values,
        )

    def enforce_boundary_flux(
        self,
        face_flux: TensorLike,
    ) -> TensorLike:
        """Set resolved velocity flux on Dirichlet boundary faces."""
        faces = self.boundary.dirichlet_operator.faces
        values = self.boundary.dirichlet_face_values
        target = bm.einsum(
            "ij,ij->i",
            values,
            self.discretization.geometry.S_f[faces],
        )
        return bm.set_at(face_flux, faces, target)

    def compute_flux(self, face_velocity: TensorLike) -> TensorLike:
        """Return owner-oriented integrated flux ``dot(U_f, S_f)``."""
        return bm.einsum(
            "ij,ij->i",
            face_velocity,
            self.discretization.geometry.S_f,
        )

    def enforce_flux(
        self,
        face_velocity: TensorLike,
        current_flux: TensorLike,
        target_flux: TensorLike,
    ) -> TensorLike:
        """Change only the normal face-velocity component to match flux."""
        surface_vector = self.discretization.geometry.S_f
        surface_norm_sq = bm.einsum(
            "ij,ij->i",
            surface_vector,
            surface_vector,
        )
        return face_velocity + (
            (target_flux - current_flux) / surface_norm_sq
        )[:, None] * surface_vector


__all__ = [
    "CollocatedSpatialFaceVelocity",
    "reconstruct_second_order_face_velocity",
]
