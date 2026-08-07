"""Cashocs-style geometric regularization for obstacle shape optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fealpy.backend import backend_manager as bm


def _resolve_design_boundary_node_order(
    current_state: Any,
    objective_parameters: Any,
    fallback_order: Sequence[int],
) -> tuple[int, ...]:
    for source in (current_state, objective_parameters):
        order = None
        if isinstance(source, Mapping):
            order = source.get("design_boundary_node_order")
            if order is None:
                order = source.get("design_boundary_node_ids")
        else:
            order = getattr(source, "design_boundary_node_order", None)
            if order is None:
                order = getattr(source, "design_boundary_node_ids", None)
        if order is not None:
            return tuple(int(node) for node in bm.asarray(order, dtype=int).reshape(-1).tolist())
    return tuple(int(node) for node in bm.asarray(tuple(fallback_order), dtype=int).reshape(-1).tolist())


def _polygon_area_and_centroid(points: Any, fallback_center: tuple[float, float]) -> tuple[float, Any]:
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
        return 0.0, bm.asarray(fallback_center, dtype=float)

    x = coords[:, 0]
    y = coords[:, 1]
    x_next = bm.roll(x, -1)
    y_next = bm.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(bm.sum(cross))
    if abs(signed_area) <= 1.0e-14:
        return 0.0, bm.mean(coords, axis=0)

    cx = float(bm.sum((x + x_next) * cross) / (6.0 * signed_area))
    cy = float(bm.sum((y + y_next) * cross) / (6.0 * signed_area))
    return abs(signed_area), bm.asarray([cx, cy], dtype=float)


def _polygon_area_moment_gradients(points: Any) -> tuple[float, Any, Any, Any, Any]:
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
        empty = bm.zeros_like(coords, dtype=float)
        return 0.0, bm.zeros(2, dtype=float), empty, empty, empty

    x = coords[:, 0]
    y = coords[:, 1]
    x_prev = bm.roll(x, 1)
    x_next = bm.roll(x, -1)
    y_prev = bm.roll(y, 1)
    y_next = bm.roll(y, -1)

    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(bm.sum(cross))
    signed_moment_x = (1.0 / 6.0) * float(bm.sum((x + x_next) * cross))
    signed_moment_y = (1.0 / 6.0) * float(bm.sum((y + y_next) * cross))

    area_grad = 0.5 * bm.stack([y_next - y_prev, x_prev - x_next], axis=-1)
    cross_prev = bm.roll(cross, 1)
    moment_x_grad = (1.0 / 6.0) * bm.stack(
        [
            cross + cross_prev + (x + x_next) * y_next - (x_prev + x) * y_prev,
            (x_prev + x) * x_prev - (x + x_next) * x_next,
        ],
        axis=-1,
    )
    moment_y_grad = (1.0 / 6.0) * bm.stack(
        [
            (y + y_next) * y_next - (y_prev + y) * y_prev,
            cross + cross_prev - (y + y_next) * x_next + (y_prev + y) * x_prev,
        ],
        axis=-1,
    )
    return (
        signed_area,
        bm.asarray([signed_moment_x, signed_moment_y], dtype=float),
        area_grad,
        moment_x_grad,
        moment_y_grad,
    )


def _polygon_vertex_weights(points: Any) -> Any:
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return bm.zeros((0,), dtype=float)
    if coords.shape[0] == 1:
        return bm.ones((1,), dtype=float)
    prev_points = bm.roll(coords, 1, axis=0)
    next_points = bm.roll(coords, -1, axis=0)
    prev_lengths = bm.linalg.norm(coords - prev_points, axis=1)
    next_lengths = bm.linalg.norm(next_points - coords, axis=1)
    return 0.5 * (prev_lengths + next_lengths)


def _polygon_signed_area(points: Any) -> float:
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(bm.sum(x * bm.roll(y, -1) - bm.roll(x, -1) * y))


def _polygon_vertex_normals(points: Any, is_hole_boundary: bool | None = None) -> Any:
    coords = bm.asarray(points, dtype=float)
    if coords.shape[0] == 0:
        return bm.zeros_like(coords)
    if coords.shape[0] == 1:
        return bm.asarray([[1.0, 0.0]], dtype=float)

    centroid = bm.mean(coords, axis=0)
    prev_points = bm.roll(coords, 1, axis=0)
    next_points = bm.roll(coords, -1, axis=0)
    prev_edge = coords - prev_points
    next_edge = next_points - coords
    prev_norm = bm.linalg.norm(prev_edge, axis=1)
    next_norm = bm.linalg.norm(next_edge, axis=1)
    tangent = (
        bm.divide(prev_edge, prev_norm[:, None], out=bm.zeros_like(prev_edge), where=prev_norm[:, None] > 0.0)
        + bm.divide(next_edge, next_norm[:, None], out=bm.zeros_like(next_edge), where=next_norm[:, None] > 0.0)
    )
    tangent_norm = bm.linalg.norm(tangent, axis=1)
    tangent = bm.where(tangent_norm[:, None] > 0.0, tangent, next_points - prev_points)
    normals = bm.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    normal_norm = bm.linalg.norm(normals, axis=1)
    fallback = coords - centroid
    fallback_norm = bm.linalg.norm(fallback, axis=1)
    normals = bm.where(normal_norm[:, None] > 0.0, normals, fallback)
    normals = bm.where(fallback_norm[:, None] > 0.0, normals, bm.asarray([[1.0, 0.0]], dtype=float))
    alignment = bm.sum(normals * (coords - centroid), axis=1)
    normals = bm.where(alignment[:, None] < 0.0, -normals, normals)
    if is_hole_boundary is None:
        is_hole_boundary = _polygon_signed_area(coords) < 0.0
    if is_hole_boundary:
        normals = -normals
    
    norm = bm.linalg.norm(normals, axis=1)
    return bm.divide(normals, norm[:, None], out=bm.zeros_like(normals), where=norm[:, None] > 0.0)


@dataclass(slots=True)
class ObstacleGeometryRegularization:
    """Cashocs-style volume/barycenter penalty for the obstacle boundary."""

    design_node_order: tuple[int, ...]
    reference_volume: float
    reference_barycenter: Any
    factor_volume: float
    factor_barycenter: float
    current_volume: float | None = None
    current_barycenter: Any | None = None
    current_signed_area: float | None = None

    def _resolve_design_node_order(self, current_state: Any, objective_parameters: Any) -> tuple[int, ...]:
        return _resolve_design_boundary_node_order(current_state, objective_parameters, self.design_node_order)

    def update(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> None:
        self.update_geometric_quantities(mesh, objective_parameters=objective_parameters, current_state=current_state)

    def update_geometric_quantities(
        self,
        mesh: Any,
        *,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> None:
        nodes = mesh.node
        design_node_order = self._resolve_design_node_order(current_state, objective_parameters)
        design_ids = bm.asarray(design_node_order, dtype=int)
        if nodes is None or design_ids.size < 3:
            self.current_volume = 0.0
            self.current_barycenter = bm.zeros(2, dtype=float)
            self.current_signed_area = 0.0
            return

        coords = nodes[design_ids]
        current_volume, current_barycenter = _polygon_area_and_centroid(coords, fallback_center=tuple(coords.mean(axis=0)))
        signed_area, signed_moments, _, _, _ = _polygon_area_moment_gradients(coords)

        self.current_volume = float(current_volume)
        self.current_barycenter = bm.asarray(current_barycenter, dtype=float).reshape(-1)[:2]
        self.current_signed_area = float(signed_area) if abs(signed_area) > 1.0e-14 else 0.0
        if abs(self.current_signed_area) <= 1.0e-14:
            self.current_signed_area = 0.0

    def _current_geometry(
        self,
        mesh: Any,
        *,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> tuple[Any, Any, float, Any, Any, Any, Any, Any]:
        nodes = mesh.node
        design_node_order = self._resolve_design_node_order(current_state, objective_parameters)
        design_ids = bm.asarray(design_node_order, dtype=int)
        if nodes is None or design_ids.size < 3:
            empty = bm.zeros((0, 2), dtype=float)
            return design_ids, empty, 0.0, bm.zeros(2, dtype=float), empty, empty, empty, empty

        coords = nodes[design_ids]
        signed_area, signed_moments, area_grad, moment_x_grad, moment_y_grad = _polygon_area_moment_gradients(coords)
        if abs(signed_area) <= 1.0e-14:
            current_volume = 0.0
            current_barycenter = bm.mean(coords, axis=0)[:2]
        else:
            current_volume = abs(signed_area)
            current_barycenter = signed_moments / signed_area
        return (
            design_ids,
            coords,
            float(current_volume),
            bm.asarray(current_barycenter, dtype=float).reshape(-1)[:2],
            bm.asarray(area_grad, dtype=float),
            bm.asarray(moment_x_grad, dtype=float),
            bm.asarray(moment_y_grad, dtype=float),
            bm.asarray([signed_area, signed_moments[0], signed_moments[1]], dtype=float),
        )

    def compute_objective(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> float:
        _, _, current_volume, current_barycenter, *_ = self._current_geometry(
            mesh, objective_parameters=objective_parameters, current_state=current_state
        )
        value = 0.0
        if self.factor_volume != 0.0:
            value += 0.5 * float(self.factor_volume) * float(current_volume - self.reference_volume) ** 2
        if self.factor_barycenter != 0.0:
            diff = bm.asarray(current_barycenter, dtype=float).reshape(-1)[:2] - self.reference_barycenter[:2]
            value += 0.5 * float(self.factor_barycenter) * float(bm.dot(diff, diff))
        return float(value)

    def update_objective_state(
        self,
        mesh: Any,
        *,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> None:
        self.update_geometric_quantities(mesh, objective_parameters=objective_parameters, current_state=current_state)

    def volume_term(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> float:
        _, _, current_volume, _, _, _, _, _ = self._current_geometry(
            mesh, objective_parameters=objective_parameters, current_state=current_state
        )
        if self.factor_volume == 0.0:
            return 0.0
        return 0.5 * float(self.factor_volume) * float(current_volume - self.reference_volume) ** 2

    def barycenter_term(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> float:
        _, _, _, current_barycenter, _, _, _, _ = self._current_geometry(
            mesh, objective_parameters=objective_parameters, current_state=current_state
        )
        if self.factor_barycenter == 0.0:
            return 0.0
        diff = bm.asarray(current_barycenter, dtype=float).reshape(-1)[:2] - self.reference_barycenter[:2]
        return 0.5 * float(self.factor_barycenter) * float(bm.dot(diff, diff))

    def shape_gradient(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
    ) -> Any:
        design_ids, coords, current_volume, current_barycenter, area_grad, moment_x_grad, moment_y_grad, signed_pack = (
            self._current_geometry(mesh, objective_parameters=objective_parameters, current_state=current_state)
        )
        if design_ids.size == 0 or coords.size == 0:
            return bm.zeros((0, 2), dtype=float)

        signed_area = float(signed_pack[0])
        signed_moment_x = float(signed_pack[1])
        signed_moment_y = float(signed_pack[2])
        gradient = bm.zeros_like(coords, dtype=float)

        if abs(signed_area) <= 1.0e-14:
            return gradient

        if self.factor_volume != 0.0:
            gradient += (
                float(self.factor_volume)
                * float(current_volume - self.reference_volume)
                * (1.0 if signed_area >= 0.0 else -1.0)
                * area_grad
            )

        if self.factor_barycenter != 0.0:
            diff = bm.asarray(current_barycenter, dtype=float).reshape(-1)[:2] - self.reference_barycenter[:2]
            inv_area_sq = 1.0 / (signed_area * signed_area)
            centroid_x_grad = (moment_x_grad * signed_area - signed_moment_x * area_grad) * inv_area_sq
            centroid_y_grad = (moment_y_grad * signed_area - signed_moment_y * area_grad) * inv_area_sq
            gradient += float(self.factor_barycenter) * (
                diff[0] * centroid_x_grad + diff[1] * centroid_y_grad
            )

        return gradient

    def shape_derivative_source(
        self,
        mesh: Any,
        state_result: Any = None,
        objective_parameters: Any = None,
        current_state: Any = None,
        *,
        contributions: Mapping[str, Any] | None = None,
        total_objective: Any = None,
    ) -> Any:
        if self.factor_volume == 0.0 and self.factor_barycenter == 0.0:
            return None
        design_ids, _, _, _, _, _, _, _ = self._current_geometry(
            mesh, objective_parameters=objective_parameters, current_state=current_state
        )
        if design_ids.size == 0:
            return None
        gradient = self.shape_gradient(
            mesh,
            state_result=state_result,
            objective_parameters=objective_parameters,
            current_state=current_state,
        )
        return {
            int(node_id): (float(vector[0]), float(vector[1]))
            for node_id, vector in zip(design_ids.tolist(), gradient, strict=True)
        }
