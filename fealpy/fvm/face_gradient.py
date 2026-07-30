"""Face-gradient construction for finite-volume solvers."""

from __future__ import annotations

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry, interpolate_cell_to_face
from .gradient_reconstruct import ResolvedGradientBoundary


def reconstruct_face_gradient(
    geometry: FVMGeometry,
    cell_gradient: TensorLike,
    cell_values: TensorLike,
    *,
    interpolation_method: str = "average",
    boundary: ResolvedGradientBoundary,
) -> TensorLike:
    """Build face gradients from cell gradients and resolved patch data.

    Internal faces use the requested owner-neighbour interpolation.  Boundary
    faces are corrected by the supplied normal derivative for Dirichlet,
    Neumann, or generic boundary patch data.
    """
    if not isinstance(geometry, FVMGeometry):
        raise TypeError("geometry must be an FVMGeometry.")
    if not isinstance(boundary, ResolvedGradientBoundary):
        raise TypeError("boundary must be a ResolvedGradientBoundary.")
    face_gradient = interpolate_cell_to_face(
        cell_gradient,
        geometry=geometry,
        method=interpolation_method,
    )

    patch_sn_grads = []
    faces = boundary.dirichlet_faces
    if faces.shape[0] > 0:
        boundary_values = boundary.dirichlet_values
        expected_shape = (faces.shape[0],) + cell_values.shape[1:]
        if boundary_values.shape != expected_shape:
            raise ValueError(
                f"dirichlet_values must have shape {expected_shape} matching "
                "dirichlet_faces."
            )
        owner = geometry.owner[faces]
        distance = geometry.normal_distance(faces)
        if bm.any(distance <= 0.0):
            raise ValueError("boundary face has zero owner-normal distance.")

        distance_shape = (distance.shape[0],) + (1,) * (cell_values.ndim - 1)
        sn_grad = (boundary_values - cell_values[owner]) / distance.reshape(distance_shape)
        patch_sn_grads.append((faces, sn_grad, "dirichlet_sn_grad"))

    faces = boundary.neumann_faces
    if faces.shape[0] > 0:
        patch_sn_grads.append(
            (faces, boundary.neumann_sn_grad, "neumann_sn_grad")
        )

    for faces, sn_grad, name in patch_sn_grads:
        expected_shape = (faces.shape[0],) + cell_gradient.shape[1:-1]
        if sn_grad.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape} matching its faces."
            )
        owner_gradient = cell_gradient[geometry.owner[faces]]
        unit_normal = geometry.n_f[faces]

        if owner_gradient.ndim == 2:
            current_sn_grad = bm.einsum("fi,fi->f", owner_gradient, unit_normal)
            correction = sn_grad - current_sn_grad
            corrected_boundary = owner_gradient + correction[:, None] * unit_normal
        elif owner_gradient.ndim == 3:
            current_sn_grad = bm.einsum("fij,fj->fi", owner_gradient, unit_normal)
            correction = sn_grad - current_sn_grad
            corrected_boundary = owner_gradient + correction[:, :, None] * unit_normal[:, None, :]
        else:
            raise ValueError("cell_gradient must have shape (NC, GD) or (NC, n_component, GD).")
        face_gradient = bm.set_at(face_gradient, faces, corrected_boundary)

    return face_gradient
