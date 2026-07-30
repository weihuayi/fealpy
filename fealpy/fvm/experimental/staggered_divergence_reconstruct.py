"""Experimental staggered-grid divergence utilities."""

from fealpy.backend import backend_manager as bm

from ..fvm_geometry import FVMGeometry
from ..simple_residual import normalized_flux_residual


class StaggeredDivergenceReconstruct:
    """Divergence from scalar staggered normal velocities.

    This class is local to the experimental staggered solvers.  It assumes the
    current structured staggered pressure mesh, where each pressure face is
    axis-aligned and the signed scalar face measure can be recovered from the
    oriented face vector by summing its components.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.geometry = FVMGeometry(mesh)

    def StagReconstruct(self, edge_velocity):
        signed_face_measure = bm.sum(self.mesh.edge_normal(), axis=1)
        flux = edge_velocity * signed_face_measure
        pe2c = self.mesh.edge_to_cell()[:, :2]
        div_u = bm.zeros(self.mesh.number_of_cells(), dtype=flux.dtype)
        mask = pe2c[:, 1] != pe2c[:, 0]
        div_u = bm.index_add(div_u, pe2c[:, 0], flux, axis=0)
        return bm.index_add(div_u, pe2c[mask, 1], flux[mask], axis=0, alpha=-1)


def staggered_mass_residual(geometry, edge_velocity):
    """Mass residual for scalar staggered velocities on pressure faces."""
    mesh = geometry.mesh
    signed_face_measure = bm.sum(mesh.edge_normal(), axis=1)
    face_flux = edge_velocity * signed_face_measure
    cell_flux_imbalance = StaggeredDivergenceReconstruct(mesh).StagReconstruct(
        edge_velocity
    )
    return normalized_flux_residual(
        cell_flux_imbalance,
        face_flux,
        geometry=geometry,
    )
