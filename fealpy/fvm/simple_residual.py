"""Residual measures shared by finite-volume pressure-velocity solvers.

The functions in this module are algebraic utilities, not solver iteration
logic.  They define residual quantities shared by pressure-velocity coupling
schemes:

* cell L2 norms for cell-centred correction fields;
* normalized and maximum finite-volume mass imbalance from signed face fluxes;

Each solver owns its iteration record and stopping criterion.  In particular,
SIMPLE uses ``SimpleIterationResidual`` and ``SimpleIterationControls`` rather
than a dictionary-building compatibility helper in this module.
"""

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


def _as_float(value: TensorLike) -> float:
    """Convert backend scalar tensors to Python floats for logging/tests."""
    return float(bm.to_numpy(value))


@dataclass(frozen=True)
class MassResidualMetrics:
    """Dimensionless and dimensional finite-volume mass residuals."""

    relative_l1: float
    relative_l2: float
    divergence_l2: float
    absolute_linf: float


def cell_l2_norm(
    value: TensorLike,
    *,
    geometry: FVMGeometry,
) -> float:
    """Return the cell-volume weighted L2 norm of a cell-centred field.

    For cell values ``q_K`` this computes

        ||q||_0 = sqrt(sum_K |K| q_K^2).

    The result is a dimensional norm; no domain-volume normalization is applied.
    """
    cell_measure = geometry.cell_measure
    return _as_float(bm.sqrt(bm.sum(cell_measure * value**2)))


def relative_l2_update(
    update: TensorLike,
    reference: TensorLike,
    *,
    geometry: FVMGeometry,
    floor: float = 1.0,
) -> float:
    """Return ``||update|| / (||reference|| + floor)`` for iteration control.

    The ``floor`` prevents a zero initial pressure or velocity field from
    turning the first relative update into a singular diagnostic.  It is part
    of the stopping criterion scale, not a numerical correction to the solve.
    """
    update_norm = cell_l2_norm(update, geometry=geometry)
    reference_norm = cell_l2_norm(reference, geometry=geometry)
    return update_norm / (reference_norm + floor)


def normalized_flux_residual(
    cell_flux_imbalance: TensorLike,
    face_flux: TensorLike,
    *,
    geometry: FVMGeometry,
    eps: float = 1.0e-30,
) -> float:
    """Return a dimensionless finite-volume continuity residual.

    ``cell_flux_imbalance`` is the cell residual

        r_K = sum_{f in dK} phi_f,

    with signed outward flux convention.  The numerator is ``sum_K |r_K|``.
    The denominator is the absolute face-flux scale with internal faces counted
    twice and boundary faces once, matching how each face contributes to cell
    balances.  A value near zero means that the supplied face fluxes satisfy
    discrete incompressibility relative to their own flux scale.
    """
    is_internal = geometry.is_internal
    numerator = _as_float(bm.sum(bm.abs(cell_flux_imbalance)))
    denominator = bm.sum(bm.abs(face_flux))
    denominator = denominator + bm.sum(bm.abs(face_flux[is_internal]))
    denominator = _as_float(denominator)

    if denominator <= eps:
        return 0.0 if numerator <= eps else numerator
    return numerator / denominator


def collocated_mass_residual(
    face_velocity: TensorLike,
    *,
    geometry: FVMGeometry,
) -> float:
    """Mass residual for collocated vector face velocities.

    The scalar flux is ``phi_f = dot(u_f, S_f)`` where ``S_f`` is the oriented
    face area vector.  The cell imbalance is obtained by scattering this same
    face flux with the owner-oriented geometry convention.
    """
    return collocated_mass_metrics(
        face_velocity,
        geometry=geometry,
    ).relative_l2


def collocated_mass_metrics(
    face_velocity: TensorLike,
    *,
    geometry: FVMGeometry,
) -> MassResidualMetrics:
    """Return mesh-scaled continuity metrics for face velocities."""
    face_flux = bm.einsum("ij,ij->i", face_velocity, geometry.S_f)
    cell_flux_imbalance = geometry.scatter_face_flux_to_cells(face_flux)
    cell_measure = geometry.cell_measure
    divergence_l2 = _as_float(
        bm.sqrt(bm.sum(cell_flux_imbalance**2 / cell_measure))
    )

    absolute_face_flux = bm.abs(face_flux)
    cell_throughput = bm.zeros_like(cell_flux_imbalance)
    cell_throughput = bm.index_add(
        cell_throughput,
        geometry.owner,
        absolute_face_flux,
        axis=0,
    )
    cell_throughput = bm.index_add(
        cell_throughput,
        geometry.neighbour[geometry.is_internal],
        absolute_face_flux[geometry.is_internal],
        axis=0,
    )
    throughput_l2 = _as_float(
        bm.sqrt(bm.sum(cell_throughput**2 / cell_measure))
    )
    if throughput_l2 <= 1.0e-30:
        relative_l2 = 0.0 if divergence_l2 <= 1.0e-30 else divergence_l2
    else:
        relative_l2 = divergence_l2 / throughput_l2
    return MassResidualMetrics(
        relative_l1=normalized_flux_residual(
            cell_flux_imbalance,
            face_flux,
            geometry=geometry,
        ),
        relative_l2=relative_l2,
        divergence_l2=divergence_l2,
        absolute_linf=_as_float(
            bm.max(bm.abs(cell_flux_imbalance))
        ),
    )


__all__ = [
    "MassResidualMetrics",
    "cell_l2_norm",
    "collocated_mass_metrics",
    "collocated_mass_residual",
    "normalized_flux_residual",
    "relative_l2_update",
]
