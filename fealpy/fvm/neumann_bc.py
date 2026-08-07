"""Finite-volume Neumann boundary-condition application helpers."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


class NeumannBC:
    """Apply prescribed normal flux data to FVM algebraic systems.

    The reusable path is ``apply_diffusion(f)``, which adds the integrated
    boundary flux contribution to owner cells.
    """

    def __init__(
        self,
        geometry: FVMGeometry,
        faces: TensorLike,
        face_values: TensorLike,
    ):
        """Store explicit normal-flux data on selected boundary faces."""
        self.geometry = geometry
        self.faces = faces
        self.face_values = face_values

    def apply_diffusion(self, f: TensorLike) -> TensorLike:
        """Add integrated Neumann fluxes to owner-cell RHS entries.

        ``face_values`` is the outward normal derivative or flux density on
        ``faces``.  The finite-volume contribution is ``face_values * |S_f|``
        scattered to the boundary owner cells.
        """
        geometry = self.geometry
        boundary_integrator = (
            self.face_values * geometry.mag_S_f[self.faces]
        )
        boundary_integrator = bm.array(
            boundary_integrator,
            dtype=f.dtype,
            device=bm.get_device(f),
        )
        f = bm.index_add(
            f,
            geometry.owner[self.faces],
            boundary_integrator,
            axis=0,
        )
        return f
