"""Discrete pressure-flux primitives for collocated incompressible FVM."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike

from .collocated_discretization import CollocatedDiscretization
from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import (
    DiffusionFaceDecomposition,
    interpolate_cell_to_face,
)
from .gradient_reconstruct import ResolvedGradientBoundary
from .scalar_diffusion_integrator import ScalarDiffusionMatrixAssembler


class PressureGaugeMatrixAssembler:
    """Assemble a pressure Laplacian with a volume-weighted gauge row."""

    def __init__(
        self,
        *,
        discretization: CollocatedDiscretization,
        diffusion_method: str,
        diffusion_nonorthogonal_eps: float,
    ) -> None:
        geometry = discretization.geometry
        nc = discretization.NC
        self.sparse_shape = (nc + 1, nc + 1)
        self.pressure_diffusion = ScalarDiffusionMatrixAssembler(
            discretization.space,
            geometry=geometry,
            method=diffusion_method,
            nonorthogonal_eps=diffusion_nonorthogonal_eps,
        )

        base_counts = (
            self.pressure_diffusion.crow[1:]
            - self.pressure_diffusion.crow[:-1]
        )
        base_rows = bm.repeat(
            bm.arange(
                nc,
                dtype=self.pressure_diffusion.col.dtype,
                device=bm.get_device(self.pressure_diffusion.col),
            ),
            base_counts,
        )
        cell = bm.arange(
            nc,
            dtype=base_rows.dtype,
            device=bm.get_device(base_rows),
        )
        gauge_col = bm.full(
            (nc,),
            nc,
            dtype=base_rows.dtype,
            device=bm.get_device(base_rows),
        )
        rows = bm.concatenate([base_rows, cell, gauge_col])
        cols = bm.concatenate(
            [self.pressure_diffusion.col, gauge_col, cell]
        )
        self.gauge_values = bm.concatenate(
            [geometry.cell_measure, geometry.cell_measure]
        )

        nrow, ncol = self.sparse_shape
        flat = bm.astype(rows, bm.int64) * ncol + bm.astype(
            cols,
            bm.int64,
        )
        order = bm.argsort(flat)
        flat_sorted = flat[order]
        group_start = bm.ones(
            (flat.shape[0],),
            dtype=bm.bool,
            device=bm.get_device(flat),
        )
        group_start = bm.set_at(
            group_start,
            slice(1, None),
            flat_sorted[1:] != flat_sorted[:-1],
        )
        unique_flat = flat_sorted[group_start]
        group_id_sorted = bm.cumsum(group_start, axis=0) - 1
        self.entry_to_value = group_id_sorted[bm.argsort(order)]

        row = unique_flat // ncol
        col = unique_flat % ncol
        counts = bm.astype(
            bm.bincount(row, minlength=nrow),
            base_rows.dtype,
        )
        self.crow = bm.concatenate(
            [
                bm.zeros(
                    (1,),
                    dtype=base_rows.dtype,
                    device=bm.get_device(base_rows),
                ),
                bm.cumsum(counts, axis=0),
            ],
            axis=0,
        )
        self.col = bm.astype(col, base_rows.dtype)

    def assembly(self, coefficient: TensorLike) -> CSRTensor:
        pressure_matrix = self.pressure_diffusion.assembly(coefficient)
        local_values = bm.concatenate(
            [pressure_matrix.values, self.gauge_values]
        )
        values = bm.zeros(
            (self.col.shape[0],),
            dtype=local_values.dtype,
            device=bm.get_device(local_values),
        )
        values = bm.index_add(
            values,
            self.entry_to_value,
            local_values,
            axis=0,
        )
        return CSRTensor(
            self.crow,
            self.col,
            values,
            spshape=self.sparse_shape,
        )


class CollocatedPressureEquation:
    """Own the shared pressure Laplacian pattern and flux primitives."""

    def __init__(
        self,
        *,
        discretization: CollocatedDiscretization,
        diffusion_method: str,
        diffusion_nonorthogonal_eps: float,
        response_interpolation: str,
    ) -> None:
        self.discretization = discretization
        self.diffusion_method = diffusion_method
        self.diffusion_nonorthogonal_eps = (
            diffusion_nonorthogonal_eps
        )
        self.response_interpolation = response_interpolation
        self._diffusion_matrix_assembler = None
        self._gauge_matrix_assembler = None
        self._has_nonorthogonal_cross_flux = None

    def face_response_coefficient(
        self,
        cell_diagonal: TensorLike,
    ) -> TensorLike:
        """Interpolate the scalar cell response to every face."""
        return interpolate_cell_to_face(
            self.discretization.cell_response(cell_diagonal),
            geometry=self.discretization.geometry,
            method=self.response_interpolation,
        )

    def diffusion_decomposition(self) -> DiffusionFaceDecomposition:
        return self.discretization.geometry.diffusion_face_decomposition(
            self.diffusion_method,
            eps=self.diffusion_nonorthogonal_eps,
        )

    def has_nonorthogonal_cross_flux(self) -> bool:
        """Return whether the fixed mesh admits a nonzero cross flux."""
        cached = self._has_nonorthogonal_cross_flux
        if cached is None:
            if self.diffusion_method == "uncorrected":
                cached = False
            else:
                decomposition = self.diffusion_decomposition()
                cached = (
                    float(
                        bm.to_numpy(
                            bm.max(bm.abs(decomposition.T_f))
                        )
                    )
                    > 0.0
                )
            self._has_nonorthogonal_cross_flux = cached
        return cached

    def orthogonal_flux(
        self,
        pressure: TensorLike,
        response_coefficient: TensorLike,
    ) -> TensorLike:
        decomposition = self.diffusion_decomposition()
        coefficient = (
            response_coefficient
            * decomposition.orthogonal_factor
        )
        geometry = self.discretization.geometry
        owner = geometry.owner
        neighbour = geometry.neighbour
        return coefficient * (
            pressure[owner] - pressure[neighbour]
        )

    def nonorthogonal_cross_flux(
        self,
        pressure: TensorLike,
        response_coefficient: TensorLike,
        *,
        pressure_gradient: TensorLike,
        gradient_boundary: ResolvedGradientBoundary,
        interpolation_method: str,
    ) -> TensorLike:
        if not self.has_nonorthogonal_cross_flux():
            return bm.zeros_like(response_coefficient)
        geometry = self.discretization.geometry
        decomposition = self.diffusion_decomposition()
        face_gradient = reconstruct_face_gradient(
            geometry,
            pressure_gradient,
            pressure,
            interpolation_method=interpolation_method,
            boundary=gradient_boundary,
        )
        cross_flux = response_coefficient * bm.einsum(
            "ij,ij->i",
            decomposition.T_f,
            face_gradient,
        )
        active_faces = bm.copy(geometry.is_internal)
        active_faces = bm.set_at(
            active_faces,
            gradient_boundary.dirichlet_faces,
            True,
        )
        return bm.where(active_faces, cross_flux, 0.0)

    def add_dirichlet_flux(
        self,
        flux: TensorLike,
        pressure: TensorLike,
        response_coefficient: TensorLike,
        boundary_faces: TensorLike,
        boundary_values: TensorLike,
    ) -> TensorLike:
        decomposition = self.diffusion_decomposition()
        coefficient = (
            response_coefficient[boundary_faces]
            * decomposition.orthogonal_factor[boundary_faces]
        )
        owner = self.discretization.geometry.owner[boundary_faces]
        boundary_flux = coefficient * (
            pressure[owner] - boundary_values
        )
        return bm.set_at(
            flux,
            boundary_faces,
            flux[boundary_faces] + boundary_flux,
        )

    def diffusion_matrix(self, coefficient: TensorLike) -> CSRTensor:
        assembler = self._diffusion_matrix_assembler
        if assembler is None:
            assembler = ScalarDiffusionMatrixAssembler(
                self.discretization.space,
                geometry=self.discretization.geometry,
                method=self.diffusion_method,
                nonorthogonal_eps=(
                    self.diffusion_nonorthogonal_eps
                ),
            )
            self._diffusion_matrix_assembler = assembler
        return assembler.assembly(coefficient)

    def gauge_matrix(self, coefficient: TensorLike) -> CSRTensor:
        assembler = self._gauge_matrix_assembler
        if assembler is None:
            assembler = PressureGaugeMatrixAssembler(
                discretization=self.discretization,
                diffusion_method=self.diffusion_method,
                diffusion_nonorthogonal_eps=(
                    self.diffusion_nonorthogonal_eps
                ),
            )
            self._gauge_matrix_assembler = assembler
        return assembler.assembly(coefficient)

    def divergence_from_flux(self, face_flux: TensorLike) -> TensorLike:
        return self.discretization.geometry.scatter_face_flux_to_cells(
            face_flux
        )

    def divergence_from_face_velocity(
        self,
        face_velocity: TensorLike,
    ) -> TensorLike:
        flux = bm.einsum(
            "ij,ij->i",
            face_velocity,
            self.discretization.geometry.S_f,
        )
        return self.divergence_from_flux(flux)

    def project_rhs_to_range(self, rhs: TensorLike) -> TensorLike:
        cell_measure = self.discretization.geometry.cell_measure
        return (
            rhs
            - bm.sum(rhs)
            * cell_measure
            / bm.sum(cell_measure)
        )

    def zero_mean_pressure(self, pressure: TensorLike) -> TensorLike:
        cell_measure = self.discretization.geometry.cell_measure
        return (
            pressure
            - bm.sum(cell_measure * pressure)
            / bm.sum(cell_measure)
        )


__all__ = [
    "CollocatedPressureEquation",
    "PressureGaugeMatrixAssembler",
]
