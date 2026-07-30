"""Conservative face-flux corrections for collocated finite volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry, face_interpolation_owner_weight


def _validate_max_condition(max_condition):
    value = float(max_condition)
    if not np.isfinite(value) or value < 1.0:
        raise ValueError("max_condition must be finite and at least 1.")
    return value


@dataclass(frozen=True)
class _CellAnchoredCache:
    stencil: TensorLike
    coefficient_operator: TensorLike
    owner_feature: TensorLike
    neighbour_feature: TensorLike
    rank: np.ndarray
    condition: np.ndarray
    layer: np.ndarray
    valid: TensorLike


class CellAnchoredQuadraticFaceFluxReconstruct:
    r"""Recover one conservative face flux from P0 cell averages.

    In each cell ``K_P``, the reconstructed polynomial is constrained to have
    cell average ``U_P``:

    .. math::
        P_P(x) = U_P + g_P\cdot r
        + \frac12 H_P:(r\otimes r-M_P),\qquad r=x-x_P.

    Only the gradient and Hessian coefficients are fitted.  Owner and
    neighbour face-average predictions are blended into one owner-oriented
    integrated flux, so local conservation is unchanged.
    """

    def __init__(
        self,
        *,
        geometry: FVMGeometry,
        quadrature_order: int = 3,
        max_stencil_layers: int = 4,
        max_condition: float = 100.0,
        rank_tolerance: float = 1.0e-12,
    ) -> None:
        if quadrature_order < 2:
            raise ValueError("quadrature_order must integrate quadratic moments.")
        if max_stencil_layers < 1:
            raise ValueError("max_stencil_layers must be positive.")
        if not 0.0 < rank_tolerance < 1.0:
            raise ValueError("rank_tolerance must lie in (0, 1).")
        self.geometry = geometry
        self.GD = self.geometry.cell_center.shape[1]
        self.quadrature_order = int(quadrature_order)
        self.max_stencil_layers = int(max_stencil_layers)
        self.max_condition = _validate_max_condition(max_condition)
        self.rank_tolerance = float(rank_tolerance)
        self.ncoeff = self.GD + self.GD * (self.GD + 1) // 2
        self._cache = None

    @staticmethod
    def _as_numpy(value):
        return np.asarray(bm.to_numpy(value))

    def _central_second_moment(self, entity, center):
        def second_moment(points, entity_slice):
            delta = points - center[entity_slice][:, None, :]
            return bm.einsum("...i,...j->...ij", delta, delta)

        if entity == "cell":
            integral = self.geometry.cell_integral(
                second_moment, q=self.quadrature_order
            )
            measure = self.geometry.cell_measure
        elif entity == "face":
            integral = self.geometry.face_integral(
                second_moment, q=self.quadrature_order
            )
            measure = self.geometry.face_measure
        else:
            raise ValueError("entity must be 'cell' or 'face'.")
        return integral / measure[:, None, None]

    def _feature(self, displacement, moment_difference, scale):
        scale = np.asarray(scale, dtype=float)
        displacement = np.asarray(displacement, dtype=float) / scale[..., None]
        moment = (
            np.asarray(moment_difference, dtype=float)
            / scale[..., None, None] ** 2
        )
        columns = [displacement[..., i] for i in range(self.GD)]
        for i in range(self.GD):
            for j in range(i, self.GD):
                value = displacement[..., i] * displacement[..., j] + moment[..., i, j]
                columns.append(0.5 * value if i == j else value)
        return np.stack(columns, axis=-1)

    def _cell_adjacency(self):
        nc = self.geometry.NC
        adjacency = [set() for _ in range(nc)]
        owner = self._as_numpy(self.geometry.owner).astype(np.int64)
        neighbour = self._as_numpy(self.geometry.neighbour).astype(np.int64)
        internal = self._as_numpy(self.geometry.is_internal).astype(bool)
        for first, second in zip(owner[internal], neighbour[internal]):
            adjacency[first].add(int(second))
            adjacency[second].add(int(first))
        return adjacency

    def _matrix_rank_and_condition(self, matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular.size == 0 or singular[0] == 0.0:
            return 0, np.inf
        threshold = self.rank_tolerance * singular[0]
        rank = int(np.count_nonzero(singular > threshold))
        condition = (
            float(singular[0] / singular[-1])
            if rank == self.ncoeff
            else np.inf
        )
        return rank, condition

    def _cell_stencil(self, cell, adjacency, cell_center, cell_moment):
        visited = {cell}
        frontier = {cell}
        best_full_rank_record = None
        best_rank_deficient_record = None
        for layer in range(1, self.max_stencil_layers + 1):
            next_frontier = set()
            for current in frontier:
                next_frontier.update(adjacency[current])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            stencil = np.asarray(sorted(visited - {cell}), dtype=np.int64)
            if stencil.size:
                displacement = cell_center[stencil] - cell_center[cell]
                distance = np.linalg.norm(displacement, axis=-1)
                scale = float(np.max(distance)) if distance.size else 0.0
                if scale > 0.0:
                    feature = self._feature(
                        displacement,
                        cell_moment[stencil] - cell_moment[cell],
                        scale,
                    )
                    scaled_distance = distance / scale
                    sample_weight = 1.0 / np.maximum(scaled_distance, 0.1) ** 2
                    weighted = np.sqrt(sample_weight)[:, None] * feature
                    rank, condition = self._matrix_rank_and_condition(weighted)
                    record = (
                        stencil,
                        feature,
                        sample_weight,
                        scale,
                        rank,
                        condition,
                        layer,
                    )
                    if rank == self.ncoeff:
                        if (
                            np.isfinite(condition)
                            and condition <= self.max_condition
                        ):
                            return record
                        if (
                            best_full_rank_record is None
                            or condition <= best_full_rank_record[5]
                        ):
                            best_full_rank_record = record
                    elif (
                        best_rank_deficient_record is None
                        or rank >= best_rank_deficient_record[4]
                    ):
                        best_rank_deficient_record = record
            if not frontier:
                break
        if best_full_rank_record is not None:
            return best_full_rank_record
        if best_rank_deficient_record is not None:
            return best_rank_deficient_record
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, self.ncoeff), dtype=float),
            np.empty((0,), dtype=float),
            1.0,
            0,
            np.inf,
            0,
        )

    def _build_cache(self):
        cell_center = self._as_numpy(self.geometry.cell_center)
        face_center = self._as_numpy(self.geometry.face_center)
        cell_moment_tensor = self._central_second_moment(
            "cell",
            self.geometry.cell_center,
        )
        face_moment_tensor = self._central_second_moment(
            "face",
            self.geometry.face_center,
        )
        cell_moment = self._as_numpy(cell_moment_tensor)
        face_moment = self._as_numpy(face_moment_tensor)
        adjacency = self._cell_adjacency()
        nc = self.geometry.NC
        records = [
            self._cell_stencil(cell, adjacency, cell_center, cell_moment)
            for cell in range(nc)
        ]

        maximum_size = max(record[0].size for record in records)
        stencil = np.broadcast_to(
            np.arange(nc, dtype=np.int64)[:, None],
            (nc, maximum_size),
        ).copy()
        coefficient_operator = np.zeros(
            (nc, self.ncoeff, maximum_size),
            dtype=float,
        )
        rank = np.empty(nc, dtype=np.int64)
        condition = np.empty(nc, dtype=float)
        layer = np.empty(nc, dtype=np.int64)
        valid = np.empty(nc, dtype=bool)
        characteristic = np.empty(nc, dtype=float)

        for cell, record in enumerate(records):
            cells, feature, sample_weight, scale, cell_rank, cell_condition, cell_layer = record
            stencil[cell, : cells.size] = cells
            cell_valid = (
                cell_rank == self.ncoeff
                and np.isfinite(cell_condition)
                and cell_condition <= self.max_condition
            )
            if cell_valid:
                weighted = np.sqrt(sample_weight)[:, None] * feature
                operator = np.linalg.pinv(
                    weighted,
                    rcond=self.rank_tolerance,
                ) * np.sqrt(sample_weight)[None, :]
                coefficient_operator[cell, :, : cells.size] = operator
            rank[cell] = cell_rank
            condition[cell] = cell_condition
            layer[cell] = cell_layer
            valid[cell] = cell_valid
            characteristic[cell] = scale

        owner = self._as_numpy(self.geometry.owner).astype(np.int64)
        neighbour = self._as_numpy(self.geometry.neighbour).astype(np.int64)
        owner_feature = self._feature(
            face_center - cell_center[owner],
            face_moment - cell_moment[owner],
            characteristic[owner],
        )
        neighbour_feature = self._feature(
            face_center - cell_center[neighbour],
            face_moment - cell_moment[neighbour],
            characteristic[neighbour],
        )

        dtype = self.geometry.cell_center.dtype
        float_device = bm.get_device(self.geometry.cell_center)
        index_device = bm.get_device(self.geometry.owner)
        self._cache = _CellAnchoredCache(
            stencil=bm.array(
                stencil,
                dtype=self.geometry.owner.dtype,
                device=index_device,
            ),
            coefficient_operator=bm.array(
                coefficient_operator,
                dtype=dtype,
                device=float_device,
            ),
            owner_feature=bm.array(
                owner_feature,
                dtype=dtype,
                device=float_device,
            ),
            neighbour_feature=bm.array(
                neighbour_feature,
                dtype=dtype,
                device=float_device,
            ),
            rank=rank,
            condition=condition,
            layer=layer,
            valid=bm.array(
                valid,
                dtype=bm.bool,
                device=index_device,
            ),
        )

    def clear_cache(self):
        """Drop geometry-dependent reconstruction data."""
        self._cache = None

    def _coefficients(self, cell_velocity):
        if self._cache is None:
            self._build_cache()
        delta = cell_velocity[self._cache.stencil] - cell_velocity[:, None, :]
        return bm.einsum(
            "nks,nsc->nkc",
            self._cache.coefficient_operator,
            delta,
        )

    def face_average(
        self,
        cell_velocity: TensorLike,
        boundary_face_average: TensorLike,
        boundary_faces: TensorLike,
    ) -> TensorLike:
        """Return one vector area average per face."""
        expected = (self.geometry.NC, self.GD)
        if cell_velocity.ndim != 2 or cell_velocity.shape != expected:
            raise ValueError("cell_velocity must have shape (NC, GD).")
        coefficients = self._coefficients(cell_velocity)
        owner = self.geometry.owner
        neighbour = self.geometry.neighbour
        owner_value = cell_velocity[owner] + bm.einsum(
            "fk,fkc->fc",
            self._cache.owner_feature,
            coefficients[owner],
        )
        neighbour_value = cell_velocity[neighbour] + bm.einsum(
            "fk,fkc->fc",
            self._cache.neighbour_feature,
            coefficients[neighbour],
        )
        owner_weight = face_interpolation_owner_weight(
            self.geometry,
            method="linear",
        )
        result = (
            owner_weight[:, None] * owner_value
            + (1.0 - owner_weight[:, None]) * neighbour_value
        )
        if boundary_face_average.shape != (
            boundary_faces.shape[0],
            self.GD,
        ):
            raise ValueError(
                "boundary_face_average must have shape "
                "(number_of_selected_boundary_faces, GD)."
            )
        result = bm.set_at(
            result,
            boundary_faces,
            boundary_face_average,
        )
        return result

    def reconstruct(
        self,
        cell_velocity: TensorLike,
        boundary_face_average: TensorLike,
        boundary_faces: TensorLike,
    ) -> TensorLike:
        """Return one owner-oriented integrated velocity flux per face."""
        face_average = self.face_average(
            cell_velocity,
            boundary_face_average=boundary_face_average,
            boundary_faces=boundary_faces,
        )
        return bm.einsum("fi,fi->f", face_average, self.geometry.S_f)

    def correction(
        self,
        cell_velocity: TensorLike,
        base_face_velocity: TensorLike,
        boundary_face_average: TensorLike,
        boundary_faces: TensorLike,
    ) -> TensorLike:
        """Return the integrated flux defect relative to a base face field.

        Faces touching a rank-deficient or ill-conditioned cell receive zero
        defect, so the caller's already-defined base flux remains active there.
        """
        target = self.reconstruct(
            cell_velocity,
            boundary_face_average=boundary_face_average,
            boundary_faces=boundary_faces,
        )
        base = bm.einsum("fi,fi->f", base_face_velocity, self.geometry.S_f)
        valid_cell = self._cache.valid
        valid_face = (
            valid_cell[self.geometry.owner]
            & valid_cell[self.geometry.neighbour]
        )
        return bm.where(valid_face, target - base, bm.zeros_like(base))

    def diagnostics(self):
        """Return rank and conditioning data for the cached cell stencils."""
        if self._cache is None:
            self._build_cache()
        valid_cell = self._as_numpy(self._cache.valid).astype(bool)
        rank_deficient_cell = self._cache.rank < self.ncoeff
        ill_conditioned_cell = ~rank_deficient_cell & ~valid_cell
        rank_deficient_cell_count = int(np.count_nonzero(rank_deficient_cell))
        ill_conditioned_cell_count = int(np.count_nonzero(ill_conditioned_cell))
        fallback_cell_count = rank_deficient_cell_count + ill_conditioned_cell_count
        owner = self._as_numpy(self.geometry.owner).astype(np.int64)
        neighbour = self._as_numpy(self.geometry.neighbour).astype(np.int64)
        fallback_face_count = np.count_nonzero(
            ~(valid_cell[owner] & valid_cell[neighbour])
        )
        return {
            "minimum_rank": int(np.min(self._cache.rank)),
            "condition_limit": self.max_condition,
            "rank_deficient_cell_count": rank_deficient_cell_count,
            "ill_conditioned_cell_count": ill_conditioned_cell_count,
            "fallback_cell_count": fallback_cell_count,
            "failed_cell_count": fallback_cell_count,
            "fallback_reason_counts": {
                "rank_deficient": rank_deficient_cell_count,
                "ill_conditioned": ill_conditioned_cell_count,
            },
            "maximum_condition": float(np.max(self._cache.condition)),
            "maximum_accepted_condition": (
                float(np.max(self._cache.condition[valid_cell]))
                if np.any(valid_cell)
                else None
            ),
            "maximum_stencil_layer": int(np.max(self._cache.layer)),
            "fallback_face_count": int(fallback_face_count),
            "stencil_layer_counts": {
                int(value): int(np.count_nonzero(self._cache.layer == value))
                for value in np.unique(self._cache.layer)
            },
        }


class FaceFluxReconstruct:
    """Variant-managed correction from cell averages to face-integrated flux."""

    def __init__(
        self,
        *,
        geometry: FVMGeometry,
        method="none",
        quadrature_order=3,
        max_stencil_layers=4,
        max_condition=100.0,
    ) -> None:
        self.geometry = geometry
        max_condition = _validate_max_condition(max_condition)
        self.cell_anchored_quadratic = None
        if method == "cell_anchored_quadratic":
            self.cell_anchored_quadratic = CellAnchoredQuadraticFaceFluxReconstruct(
                geometry=self.geometry,
                quadrature_order=quadrature_order,
                max_stencil_layers=max_stencil_layers,
                max_condition=max_condition,
            )
        if method not in self.correction:
            raise ValueError(f"Unknown face-flux correction variant: {method!r}.")
        self.method = method
        self.correction.set(method)

    @variantmethod("none")
    def correction(
        self,
        cell_velocity: TensorLike,
        base_face_velocity: TensorLike,
        boundary_face_average: TensorLike,
        boundary_faces: TensorLike,
    ) -> TensorLike:
        return bm.zeros_like(self.geometry.mag_S_f)

    @correction.register("cell_anchored_quadratic")
    def correction(
        self,
        cell_velocity: TensorLike,
        base_face_velocity: TensorLike,
        boundary_face_average: TensorLike,
        boundary_faces: TensorLike,
    ) -> TensorLike:
        return self.cell_anchored_quadratic.correction(
            cell_velocity,
            base_face_velocity,
            boundary_face_average=boundary_face_average,
            boundary_faces=boundary_faces,
        )

    def diagnostics(self):
        """Return diagnostics for the active correction variant."""
        if self.cell_anchored_quadratic is None:
            return {"method": "none"}
        result = self.cell_anchored_quadratic.diagnostics()
        result["method"] = self.method
        return result

    def clear_cache(self):
        """Drop active geometry-dependent caches."""
        if self.cell_anchored_quadratic is not None:
            self.cell_anchored_quadratic.clear_cache()
