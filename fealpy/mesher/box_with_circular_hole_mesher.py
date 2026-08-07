from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..backend import backend_manager as bm
from ..mesh import TriangleMesh

try:
    import gmsh
except ImportError:
    raise ImportError(
        "The gmsh package is required for the box-with-circular-hole mesher. Please install it via 'pip install gmsh'."
    )


class BoxWithCircularHoleMesher2D:
    """2D box-with-circular-hole mesher with boundary metadata attached to the mesh."""

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        return {
            "box": (-3.0, 3.0, -2.0, 2.0),
            "center": (0.0, 0.0),
            "radius": 0.5,
            "h": 0.375,
            "mesh_size_profile": "uniform",
            "mesh_size_inner": None,
            "mesh_size_outer": None,
            "mesh_size_transition": None,
        }

    @classmethod
    def validate_parameters(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        defaults = cls.default_parameters()
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            raise ValueError(f"unknown parameter keys: {unknown}")
        raw = {**defaults, **dict(params)}
        box = tuple(float(v) for v in raw["box"])
        center = tuple(float(v) for v in raw["center"])
        radius = float(raw["radius"])
        h = float(raw["h"])
        if len(box) != 4:
            raise ValueError("box must contain four values")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if h <= 0.0:
            raise ValueError("h must be positive")
        return {
            "box": box,
            "center": center,
            "radius": radius,
            "h": h,
            "mesh_size_profile": str(raw["mesh_size_profile"]),
            "mesh_size_inner": None if raw["mesh_size_inner"] is None else float(raw["mesh_size_inner"]),
            "mesh_size_outer": None if raw["mesh_size_outer"] is None else float(raw["mesh_size_outer"]),
            "mesh_size_transition": None if raw["mesh_size_transition"] is None else float(raw["mesh_size_transition"]),
        }

    def __init__(self, params: Mapping[str, Any] | None = None, gmsh_module: Any | None = None):
        self.params = self.validate_parameters(self.default_parameters() if params is None else params)
        self.gmsh = gmsh if gmsh_module is None else gmsh_module
        self._mesh_data_cache: dict[str, Any] | None = None

    @staticmethod
    def _boundary_edge_mask(mesh: Any) -> np.ndarray:
        for attribute in ("boundary_edge_flag", "is_boundary_edge"):
            method = getattr(mesh, attribute, None)
            if callable(method):
                return np.asarray(method(), dtype=bool).reshape(-1)
        raise ValueError("mesh does not expose a boundary-edge flag")

    @staticmethod
    def _sorted_boundary_nodes(node_ids: np.ndarray, nodes: np.ndarray, center: tuple[float, float]) -> np.ndarray:
        if node_ids.size == 0:
            return node_ids
        coords = nodes[np.asarray(node_ids, dtype=int)]
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        return np.asarray(node_ids, dtype=int)[np.argsort(angles)]

    def _classify_boundary_nodes_from_box(
        self,
        mesh: Any,
        box: tuple[float, float, float, float] | None = None,
        tol: float = 1.0e-10,
    ) -> dict[str, np.ndarray]:
        nodes = np.asarray(mesh.entity("node"), dtype=float)
        boundary_mask = np.zeros(mesh.number_of_nodes(), dtype=bool)
        boundary_mask[np.asarray(mesh.entity("edge"), dtype=int)[self._boundary_edge_mask(mesh)].reshape(-1)] = True
        boundary_ids = np.flatnonzero(boundary_mask)
        if boundary_ids.size == 0:
            empty = np.asarray([], dtype=int)
            return {"inlet": empty, "wall": empty, "outlet": empty, "design": empty}
        xmin, xmax, ymin, ymax = map(float, self.params["box"] if box is None else box)
        boundary_nodes = nodes[boundary_ids]
        inlet = boundary_ids[np.abs(boundary_nodes[:, 0] - xmin) <= tol]
        outlet = boundary_ids[np.abs(boundary_nodes[:, 0] - xmax) <= tol]
        wall = boundary_ids[(np.abs(boundary_nodes[:, 1] - ymin) <= tol) | (np.abs(boundary_nodes[:, 1] - ymax) <= tol)]
        outer_ids = np.asarray(np.unique(np.concatenate([inlet, wall, outlet])), dtype=int)
        design = np.setdiff1d(boundary_ids, outer_ids, assume_unique=False)
        return {
            "inlet": np.asarray(np.unique(inlet), dtype=int),
            "wall": np.asarray(np.unique(wall), dtype=int),
            "outlet": np.asarray(np.unique(outlet), dtype=int),
            "design": np.asarray(np.unique(design), dtype=int),
        }

    def _order_boundary_loop_node_ids(
        self,
        mesh: Any,
        node_ids: np.ndarray,
        *,
        fallback_center: tuple[float, float] | None = None,
    ) -> np.ndarray:
        node_ids = np.asarray(np.unique(node_ids), dtype=int)
        if node_ids.size <= 2:
            return node_ids
        nodes = np.asarray(mesh.entity("node"), dtype=float)
        boundary_edges = np.asarray(mesh.entity("edge"), dtype=int)[self._boundary_edge_mask(mesh)]
        id_set = {int(node) for node in node_ids.tolist()}
        adjacency: dict[int, list[int]] = {int(node): [] for node in node_ids.tolist()}
        for left, right in boundary_edges:
            left_id = int(left)
            right_id = int(right)
            if left_id in id_set and right_id in id_set:
                adjacency[left_id].append(right_id)
                adjacency[right_id].append(left_id)
        if any(len(neighbours) < 2 for neighbours in adjacency.values()):
            center = fallback_center if fallback_center is not None else tuple(np.mean(nodes[node_ids], axis=0).tolist())
            return self._sorted_boundary_nodes(node_ids, nodes, center)
        start = min(node_ids.tolist(), key=lambda node: (float(nodes[node, 0]), float(nodes[node, 1]), int(node)))
        ordered = [int(start)]
        previous = None
        current = int(start)
        for _ in range(node_ids.size - 1):
            neighbours = adjacency.get(current, [])
            candidates = [node for node in neighbours if node != previous]
            if not candidates:
                break
            next_node = int(candidates[0])
            if next_node == start:
                break
            ordered.append(next_node)
            previous, current = current, next_node
        if len(ordered) != node_ids.size:
            center = fallback_center if fallback_center is not None else tuple(np.mean(nodes[node_ids], axis=0).tolist())
            return self._sorted_boundary_nodes(node_ids, nodes, center)
        return np.asarray(ordered, dtype=int)

    @staticmethod
    def _polygon_area_and_centroid(points: np.ndarray, fallback_center: tuple[float, float]) -> tuple[float, np.ndarray]:
        coords = np.asarray(points, dtype=float)
        if coords.shape[0] < 3:
            return 0.0, np.asarray(fallback_center, dtype=float)
        x = coords[:, 0]
        y = coords[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        area = 0.5 * float(np.sum(cross))
        if abs(area) <= 1.0e-14:
            return 0.0, coords.mean(axis=0)
        cx = float(np.sum((x + x_next) * cross) / (6.0 * area))
        cy = float(np.sum((y + y_next) * cross) / (6.0 * area))
        return abs(area), np.array([cx, cy], dtype=float)

    def _build_box_with_circular_hole_mesh(
        self,
        box: tuple[float, float, float, float],
        center: tuple[float, float],
        radius: float,
        h: float,
        *,
        mesh_size_profile: str = "uniform",
        mesh_size_inner: float | None = None,
        mesh_size_outer: float | None = None,
        mesh_size_transition: float | None = None,
    ) -> TriangleMesh:
        normalized = str(mesh_size_profile).casefold()
        if normalized in {"cashocs", "default", "uniform"}:
            return TriangleMesh.from_box_with_circular_holes(
                box=list(box),
                holes=[(float(center[0]), float(center[1]), float(radius))],
                h=h,
            )
        if normalized != "graded":
            raise ValueError(f"Unsupported mesh_size_profile: {mesh_size_profile!r}")

        inner_size = float(mesh_size_inner if mesh_size_inner is not None else max(0.25 * h, 0.05))
        outer_size = float(mesh_size_outer if mesh_size_outer is not None else h)
        transition_distance = float(
            mesh_size_transition if mesh_size_transition is not None else max(float(radius), 2.0 * outer_size)
        )
        if inner_size <= 0.0 or outer_size <= 0.0 or transition_distance <= 0.0:
            raise ValueError("mesh size parameters must be positive")
        if inner_size > outer_size:
            raise ValueError("mesh_size_inner must be less than or equal to mesh_size_outer")

        xmin, xmax, ymin, ymax = map(float, box)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        try:
            gmsh.model.add("box_with_hole_graded")
            p1 = gmsh.model.occ.addPoint(xmin, ymin, 0.0)
            p2 = gmsh.model.occ.addPoint(xmax, ymin, 0.0)
            p3 = gmsh.model.occ.addPoint(xmax, ymax, 0.0)
            p4 = gmsh.model.occ.addPoint(xmin, ymax, 0.0)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            outer_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            circle = gmsh.model.occ.addCircle(float(center[0]), float(center[1]), 0.0, float(radius))
            hole_loop = gmsh.model.occ.addCurveLoop([circle])
            gmsh.model.occ.addPlaneSurface([outer_loop, hole_loop])
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", inner_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", outer_size)
            gmsh.option.setNumber("Mesh.Algorithm", 5)
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [circle])
            gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", inner_size)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", outer_size)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", transition_distance)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
            gmsh.model.mesh.generate(2)
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = bm.from_numpy(node_coords.reshape(-1, 3)[:, 0:2])
            tri_type = gmsh.model.mesh.getElementType("triangle", 1)
            types, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
            cell = None
            for etype, element_nodes in zip(types, elemNodeTags):
                if etype == tri_type:
                    nn = gmsh.model.mesh.getElementProperties(etype)[3]
                    cell = bm.array(element_nodes, dtype=bm.int32).reshape(-1, nn) - 1
                    break
            if cell is None:
                raise RuntimeError("graded gmsh mesh did not produce triangle cells")
        finally:
            gmsh.finalize()
        nn = len(node)
        valid = bm.zeros(nn, dtype=bm.bool)
        valid = bm.set_at(valid, cell, True)
        node = node[valid]
        idx_map = bm.zeros(nn, dtype=cell.dtype)
        idx_map = bm.set_at(idx_map, valid, bm.arange(valid.sum(), dtype=bm.int64))
        cell = idx_map[cell]
        return TriangleMesh(node, cell)

    def _attach_metadata(
        self,
        mesh: TriangleMesh,
        *,
        box: tuple[float, float, float, float],
        center: tuple[float, float],
        radius: float,
        mesh_size_profile: str,
        mesh_size_inner: float | None,
        mesh_size_outer: float | None,
        mesh_size_transition: float | None,
    ) -> TriangleMesh:
        nodes = np.asarray(mesh.entity("node"), dtype=float)
        roles = self._classify_boundary_nodes_from_box(mesh, box=box)
        design_ids = np.asarray(roles.get("design", np.asarray([], dtype=int)), dtype=int)
        design_order = self._order_boundary_loop_node_ids(mesh, design_ids, fallback_center=center) if design_ids.size else np.asarray([], dtype=int)
        if design_order.size:
            design_coords = nodes[design_order]
            design_area, design_center = self._polygon_area_and_centroid(design_coords, fallback_center=center)
            design_radius = float(np.mean(np.linalg.norm(design_coords - design_center, axis=1)))
        else:
            design_center = np.asarray(center, dtype=float)
            design_radius = float(radius)
            design_area = 0.0
        boundary_groups = [roles["inlet"], roles["wall"], roles["outlet"]]
        if design_ids.size:
            boundary_groups.append(design_ids)
        boundary_nodes_list = [np.asarray(group, dtype=int).reshape(-1) for group in boundary_groups]
        boundary_nodes_list = [group for group in boundary_nodes_list if group.size > 0]
        boundary_nodes = np.asarray(np.unique(np.concatenate(boundary_nodes_list)) if boundary_nodes_list else np.asarray([], dtype=int), dtype=int)
        fixed_nodes = np.asarray(np.unique(np.concatenate([roles["inlet"], roles["wall"], roles["outlet"]])), dtype=int)
        mesh.boundary_nodes_by_role = {
            "inlet": np.asarray(roles["inlet"], dtype=int),
            "wall": np.asarray(roles["wall"], dtype=int),
            "outlet": np.asarray(roles["outlet"], dtype=int),
            "design": np.asarray(design_ids, dtype=int),
            "fixed": np.asarray(fixed_nodes, dtype=int),
            "boundary_cycle": np.asarray(design_order, dtype=int),
        }
        mesh.inlet_nodes = np.asarray(roles["inlet"], dtype=int)
        mesh.wall_nodes = np.asarray(roles["wall"], dtype=int)
        mesh.outlet_nodes = np.asarray(roles["outlet"], dtype=int)
        mesh.design_nodes = np.asarray(design_ids, dtype=int)
        mesh.fixed_nodes = np.asarray(fixed_nodes, dtype=int)
        mesh.fixed_boundary_ids = np.asarray(fixed_nodes, dtype=int)
        mesh.boundary_nodes = np.asarray(boundary_nodes, dtype=int)
        mesh.design_boundary_ids = np.asarray(design_ids, dtype=int)
        mesh.design_boundary_node_order = tuple(int(node) for node in np.asarray(design_order, dtype=int).tolist())
        mesh.design_boundary_node_ids = tuple(int(node) for node in np.asarray(design_order, dtype=int).tolist())
        mesh.design_center = tuple(float(value) for value in np.asarray(design_center, dtype=float).reshape(-1)[:2])
        mesh.design_radius = float(design_radius)
        mesh.design_area = float(design_area)
        mesh.box = tuple(float(v) for v in box)
        mesh.obstacle_center = tuple(float(v) for v in center)
        mesh.obstacle_radius = float(radius)
        mesh.mesh_size_profile = str(mesh_size_profile)
        mesh.mesh_size_inner = mesh_size_inner
        mesh.mesh_size_outer = mesh_size_outer
        mesh.mesh_size_transition = mesh_size_transition
        return mesh

    def init_mesh(self) -> TriangleMesh:
        mesh = self._build_box_with_circular_hole_mesh(
            box=self.params["box"],
            center=self.params["center"],
            radius=self.params["radius"],
            h=self.params["h"],
            mesh_size_profile=self.params["mesh_size_profile"],
            mesh_size_inner=self.params["mesh_size_inner"],
            mesh_size_outer=self.params["mesh_size_outer"],
            mesh_size_transition=self.params["mesh_size_transition"],
        )
        return self._attach_metadata(
            mesh,
            box=self.params["box"],
            center=self.params["center"],
            radius=self.params["radius"],
            mesh_size_profile=self.params["mesh_size_profile"],
            mesh_size_inner=self.params["mesh_size_inner"],
            mesh_size_outer=self.params["mesh_size_outer"],
            mesh_size_transition=self.params["mesh_size_transition"],
        )

    def build_mesh_from_boundary_points(self, boundary_points: Any) -> TriangleMesh:
        boundary_coords = np.asarray(boundary_points, dtype=float)
        mesh = self._build_box_with_polygon_hole_mesh(
            box=self.params["box"],
            hole_points=boundary_coords,
            h=self.params["h"],
            mesh_size_profile=self.params["mesh_size_profile"],
            mesh_size_inner=self.params["mesh_size_inner"],
            mesh_size_outer=self.params["mesh_size_outer"],
            mesh_size_transition=self.params["mesh_size_transition"],
        )
        return self._attach_metadata(
            mesh,
            box=self.params["box"],
            center=self.params["center"],
            radius=self.params["radius"],
            mesh_size_profile=self.params["mesh_size_profile"],
            mesh_size_inner=self.params["mesh_size_inner"],
            mesh_size_outer=self.params["mesh_size_outer"],
            mesh_size_transition=self.params["mesh_size_transition"],
        )

    def _build_box_with_polygon_hole_mesh(
        self,
        *,
        box: tuple[float, float, float, float],
        hole_points: Any,
        h: float,
        mesh_size_profile: str = "uniform",
        mesh_size_inner: float | None = None,
        mesh_size_outer: float | None = None,
        mesh_size_transition: float | None = None,
    ) -> TriangleMesh:
        hole_coords = np.asarray(hole_points, dtype=float)
        if hole_coords.ndim != 2 or hole_coords.shape[0] < 3 or hole_coords.shape[1] < 2:
            raise ValueError("hole_points must be a polygon with at least three vertices")
        xmin, xmax, ymin, ymax = map(float, box)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        try:
            gmsh.model.add("box_with_polygon_hole")
            p1 = gmsh.model.occ.addPoint(xmin, ymin, 0.0)
            p2 = gmsh.model.occ.addPoint(xmax, ymin, 0.0)
            p3 = gmsh.model.occ.addPoint(xmax, ymax, 0.0)
            p4 = gmsh.model.occ.addPoint(xmin, ymax, 0.0)
            outer_lines = [gmsh.model.occ.addLine(p1, p2), gmsh.model.occ.addLine(p2, p3), gmsh.model.occ.addLine(p3, p4), gmsh.model.occ.addLine(p4, p1)]
            outer_loop = gmsh.model.occ.addCurveLoop(outer_lines)
            hole_points_tags = [gmsh.model.occ.addPoint(float(point[0]), float(point[1]), 0.0) for point in hole_coords]
            hole_lines = [
                gmsh.model.occ.addLine(hole_points_tags[index], hole_points_tags[(index + 1) % len(hole_points_tags)])
                for index in range(len(hole_points_tags))
            ]
            hole_loop = gmsh.model.occ.addCurveLoop(hole_lines)
            gmsh.model.occ.addPlaneSurface([outer_loop, hole_loop])
            gmsh.model.occ.synchronize()
            normalized = str(mesh_size_profile).casefold()
            if normalized in {"cashocs", "default", "uniform"}:
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
            elif normalized == "graded":
                inner_size = float(mesh_size_inner if mesh_size_inner is not None else max(0.25 * h, 0.05))
                outer_size = float(mesh_size_outer if mesh_size_outer is not None else h)
                transition_distance = float(
                    mesh_size_transition if mesh_size_transition is not None else max(0.5 * (xmax - xmin), 2.0 * outer_size)
                )
                if inner_size <= 0.0 or outer_size <= 0.0 or transition_distance <= 0.0:
                    raise ValueError("mesh size parameters must be positive")
                if inner_size > outer_size:
                    raise ValueError("mesh_size_inner must be less than or equal to mesh_size_outer")
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh.option.setNumber("Mesh.MeshSizeMin", inner_size)
                gmsh.option.setNumber("Mesh.MeshSizeMax", outer_size)
                gmsh.option.setNumber("Mesh.Algorithm", 5)
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", hole_lines)
                gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", inner_size)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", outer_size)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", transition_distance)
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
            else:
                raise ValueError(f"Unsupported mesh_size_profile: {mesh_size_profile!r}")
            gmsh.model.mesh.generate(2)
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = bm.from_numpy(node_coords.reshape(-1, 3)[:, 0:2])
            tri_type = gmsh.model.mesh.getElementType("triangle", 1)
            types, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
            cell = None
            for etype, element_nodes in zip(types, elemNodeTags):
                if etype == tri_type:
                    nn = gmsh.model.mesh.getElementProperties(etype)[3]
                    cell = bm.array(element_nodes, dtype=bm.int32).reshape(-1, nn) - 1
                    break
            if cell is None:
                raise RuntimeError("polygon-hole gmsh mesh did not produce triangle cells")
        finally:
            gmsh.finalize()
        nn = len(node)
        valid = bm.zeros(nn, dtype=bm.bool)
        valid = bm.set_at(valid, cell, True)
        node = node[valid]
        idx_map = bm.zeros(nn, dtype=cell.dtype)
        idx_map = bm.set_at(idx_map, valid, bm.arange(valid.sum(), dtype=bm.int64))
        cell = idx_map[cell]
        return TriangleMesh(node, cell)
