from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..backend import backend_manager as bm
from ..mesh import TriangleMesh
from .gmsh_fsi_pipe_mesher import build_dimtag_maps, extract_boundary_edges, extract_triangle_data

try:
    import gmsh
except ImportError:
    raise ImportError(
        "The gmsh package is required for the paper pipe mesher. Please install it via 'pip install gmsh'."
    )


DimTag = Tuple[int, int]


class PiecewiseLinearTransitionPipeMesher2D:
    """2D piecewise-linear transition pipe mesher with chamfered outer corners."""

    @staticmethod
    def default_parameters():
        return {
            "channel_width": 1.0,
            "inlet_length": 2.4,
            "bend_length": 3.0,
            "rise_height": 2.55,
            "outlet_length": 4.9,
            "corner_chamfer_ratio": 0.45,
            "design_margin_inlet_ratio": 1.0 - 1.0 / 2.4,
            "design_margin_outlet_ratio": 0.5,
            "mesh_size_global": 0.12,
            "mesh_size_profile": "graded",
            "mesh_size_inner": None,
            "mesh_size_outer": None,
            "mesh_size_transition": None,
            "line_samples_per_unit": 8.0,
            "arc_samples": 8,
        }

    @classmethod
    def validate_parameters(cls, params):
        defaults = cls.default_parameters()
        unknown_keys = sorted(set(params) - set(defaults))
        if unknown_keys:
            raise ValueError(f"unknown parameter keys: {unknown_keys}")

        raw = {**defaults, **params}
        channel_width = float(raw["channel_width"])
        inlet_length = float(raw["inlet_length"])
        bend_length = float(raw["bend_length"])
        rise_height = float(raw["rise_height"])
        outlet_length = float(raw["outlet_length"])
        corner_chamfer_ratio = float(raw["corner_chamfer_ratio"])
        design_margin_inlet_ratio = float(raw["design_margin_inlet_ratio"])
        design_margin_outlet_ratio = float(raw["design_margin_outlet_ratio"])
        mesh_size_global = float(raw["mesh_size_global"])
        mesh_size_profile = str(raw["mesh_size_profile"])
        mesh_size_inner = None if raw["mesh_size_inner"] is None else float(raw["mesh_size_inner"])
        mesh_size_outer = None if raw["mesh_size_outer"] is None else float(raw["mesh_size_outer"])
        mesh_size_transition = None if raw["mesh_size_transition"] is None else float(raw["mesh_size_transition"])
        line_samples_per_unit = float(raw["line_samples_per_unit"])
        arc_samples = int(raw["arc_samples"])

        if channel_width <= 0.0:
            raise ValueError("channel_width must be > 0")
        if inlet_length <= channel_width * 0.5:
            raise ValueError("inlet_length must be larger than half the channel width")
        if bend_length <= channel_width * 0.5:
            raise ValueError("bend_length must be larger than half the channel width")
        if rise_height <= channel_width * 0.5:
            raise ValueError("rise_height must be larger than half the channel width")
        if outlet_length <= channel_width * 0.5:
            raise ValueError("outlet_length must be larger than half the channel width")
        if not 0.0 < corner_chamfer_ratio < 1.0:
            raise ValueError("corner_chamfer_ratio must be between 0 and 1")
        if not 0.0 < design_margin_inlet_ratio < 1.0:
            raise ValueError("design_margin_inlet_ratio must be between 0 and 1")
        if not 0.0 < design_margin_outlet_ratio < 1.0:
            raise ValueError("design_margin_outlet_ratio must be between 0 and 1")
        if mesh_size_global <= 0.0:
            raise ValueError("mesh_size_global must be > 0")
        if line_samples_per_unit <= 0.0:
            raise ValueError("line_samples_per_unit must be > 0")
        if arc_samples < 3:
            raise ValueError("arc_samples must be at least 3")

        centerline_points = np.array(
            [
                [0.0, 0.0],
                [inlet_length, 0.0],
                [inlet_length + bend_length, rise_height],
                [inlet_length + bend_length + outlet_length, rise_height],
            ],
            dtype=float,
        )

        return {
            "channel_width": channel_width,
            "inlet_length": inlet_length,
            "bend_length": bend_length,
            "rise_height": rise_height,
            "outlet_length": outlet_length,
            "corner_chamfer_ratio": corner_chamfer_ratio,
            "design_margin_inlet_ratio": design_margin_inlet_ratio,
            "design_margin_outlet_ratio": design_margin_outlet_ratio,
            "mesh_size_global": mesh_size_global,
            "mesh_size_profile": mesh_size_profile,
            "mesh_size_inner": mesh_size_inner,
            "mesh_size_outer": mesh_size_outer,
            "mesh_size_transition": mesh_size_transition,
            "line_samples_per_unit": line_samples_per_unit,
            "arc_samples": arc_samples,
            "centerline_points": centerline_points,
            "fixed_x_left": inlet_length * (1.0 - design_margin_inlet_ratio),
            "fixed_x_right": inlet_length + bend_length + outlet_length * design_margin_outlet_ratio,
        }

    def __init__(self, params=None, gmsh_module=None):
        external = self.default_parameters() if params is None else dict(params)
        internal = self.validate_parameters(external)
        self.params = internal
        self.external_params = external
        self.internal_params = internal
        self.gmsh = gmsh if gmsh_module is None else gmsh_module
        self._mesh_data_cache = None

    def model_name(self) -> str:
        return "fealpy_piecewise_linear_transition_pipe_mesher_2d"

    def mesh_topology_dimension(self) -> int:
        return 2

    def geo_dimension(self) -> int:
        return 2

    @staticmethod
    def _unit_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("zero-length segment encountered in centerline")
        return vector / norm

    @staticmethod
    def _line_intersection(point_a: np.ndarray, direction_a: np.ndarray, point_b: np.ndarray, direction_b: np.ndarray) -> np.ndarray:
        matrix = np.array(
            [
                [float(direction_a[0]), -float(direction_b[0])],
                [float(direction_a[1]), -float(direction_b[1])],
            ],
            dtype=float,
        )
        rhs = np.array([float(point_b[0] - point_a[0]), float(point_b[1] - point_a[1])], dtype=float)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1.0e-14:
            return 0.5 * (np.asarray(point_a, dtype=float) + np.asarray(point_b, dtype=float))
        s, _ = np.linalg.solve(matrix, rhs)
        return np.asarray(point_a, dtype=float) + float(s) * np.asarray(direction_a, dtype=float)

    @staticmethod
    def _turn_cross(tangent_a: np.ndarray, tangent_b: np.ndarray) -> float:
        return float(tangent_a[0] * tangent_b[1] - tangent_a[1] * tangent_b[0])

    def _sample_offset_side(self, params, side_sign: float) -> np.ndarray:
        centerline = np.asarray(params["centerline_points"], dtype=float)
        if centerline.ndim != 2 or centerline.shape != (4, 2):
            raise ValueError("centerline_points must be a 4x2 array")

        half_width = 0.5 * float(params["channel_width"])
        chamfer_length = float(params["corner_chamfer_ratio"]) * float(params["channel_width"])

        segments = centerline[1:] - centerline[:-1]
        tangents = np.stack([self._unit_vector(segment) for segment in segments], axis=0)
        left_normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        side_normals = side_sign * left_normals

        points: List[np.ndarray] = [centerline[0] + half_width * side_normals[0]]
        for vertex_index in range(1, centerline.shape[0] - 1):
            prev_point = centerline[vertex_index] + half_width * side_normals[vertex_index - 1]
            next_point = centerline[vertex_index] + half_width * side_normals[vertex_index]
            turn_cross = self._turn_cross(tangents[vertex_index - 1], tangents[vertex_index])
            is_outer_corner = turn_cross * side_sign < 0.0
            if is_outer_corner:
                previous_segment = centerline[vertex_index] - centerline[vertex_index - 1]
                next_segment = centerline[vertex_index + 1] - centerline[vertex_index]
                previous_length = float(np.linalg.norm(previous_segment))
                next_length = float(np.linalg.norm(next_segment))
                local_chamfer = min(chamfer_length, 0.35 * previous_length, 0.35 * next_length)
                points.append(prev_point - local_chamfer * tangents[vertex_index - 1])
                points.append(next_point + local_chamfer * tangents[vertex_index])
            else:
                points.append(
                    self._line_intersection(
                        prev_point,
                        tangents[vertex_index - 1],
                        next_point,
                        tangents[vertex_index],
                    )
                )
        points.append(centerline[-1] + half_width * side_normals[-1])

        return np.asarray(points, dtype=float)

    def _build_boundary_polygon(self, params):
        lower = self._sample_offset_side(params, side_sign=-1.0)
        upper = self._sample_offset_side(params, side_sign=1.0)

        def _insert_x_cut(points: np.ndarray, x_cut: float) -> np.ndarray:
            coords = [np.asarray(point, dtype=float) for point in np.asarray(points, dtype=float)]
            if len(coords) < 2:
                return np.asarray(coords, dtype=float)
            for index in range(len(coords) - 1):
                x0 = float(coords[index][0])
                x1 = float(coords[index + 1][0])
                if abs(x0 - x_cut) <= 1.0e-12 or abs(x1 - x_cut) <= 1.0e-12:
                    continue
                if (x0 - x_cut) * (x1 - x_cut) > 0.0:
                    continue
                denominator = x1 - x0
                if abs(denominator) <= 1.0e-14:
                    continue
                t = (x_cut - x0) / denominator
                if t < 0.0 or t > 1.0:
                    continue
                y_cut = float(coords[index][1] + t * (coords[index + 1][1] - coords[index][1]))
                cut_point = np.array([float(x_cut), y_cut], dtype=float)
                if np.allclose(cut_point, coords[index], atol=1.0e-12, rtol=0.0) or np.allclose(cut_point, coords[index + 1], atol=1.0e-12, rtol=0.0):
                    break
                coords.insert(index + 1, cut_point)
                break
            return np.asarray(coords, dtype=float)

        fixed_x_left = float(params.get("fixed_x_left", params["inlet_length"] * (1.0 - float(params["design_margin_inlet_ratio"]))))
        fixed_x_right = float(params.get("fixed_x_right", params["inlet_length"] + params["bend_length"] + float(params["outlet_length"]) * float(params["design_margin_outlet_ratio"])))
        lower = _insert_x_cut(lower, fixed_x_left)
        lower = _insert_x_cut(lower, fixed_x_right)
        upper = _insert_x_cut(upper, fixed_x_left)
        upper = _insert_x_cut(upper, fixed_x_right)

        if lower.shape[0] < 2 or upper.shape[0] < 2:
            raise RuntimeError("failed to construct the paper pipe boundary")

        polygon = np.vstack([
            lower,
            upper[-1:],
            upper[-2::-1],
        ])
        return polygon, lower, upper

    def _build_geometry(self, gmsh_module, params):
        occ = gmsh_module.model.occ
        polygon, lower, upper = self._build_boundary_polygon(params)

        point_tags = [
            occ.addPoint(float(x), float(y), 0.0, float(params["mesh_size_global"]))
            for x, y in polygon
        ]

        line_tags = []
        for index in range(len(point_tags) - 1):
            line_tags.append(occ.addLine(point_tags[index], point_tags[index + 1]))
        line_tags.append(occ.addLine(point_tags[-1], point_tags[0]))

        loop = occ.addCurveLoop(line_tags)
        surface = occ.addPlaneSurface([loop])
        occ.synchronize()

        outlet_line_tag = line_tags[len(lower) - 1]
        inlet_line_tag = line_tags[-1]
        wall_line_tags = [
            line_tag for index, line_tag in enumerate(line_tags)
            if index not in {len(lower) - 1, len(line_tags) - 1}
        ]

        gmsh_module.model.addPhysicalGroup(2, [surface], 1)
        gmsh_module.model.setPhysicalName(2, 1, "fluid")
        gmsh_module.model.addPhysicalGroup(1, [inlet_line_tag], 2)
        gmsh_module.model.setPhysicalName(1, 2, "inlet")
        gmsh_module.model.addPhysicalGroup(1, [outlet_line_tag], 3)
        gmsh_module.model.setPhysicalName(1, 3, "outlet")
        gmsh_module.model.addPhysicalGroup(1, wall_line_tags, 4)
        gmsh_module.model.setPhysicalName(1, 4, "wall")

        physical_groups = [(2, 1, "fluid"), (1, 2, "inlet"), (1, 3, "outlet"), (1, 4, "wall")]
        physical_name_to_dimtag, physical_dimtag_to_name = build_dimtag_maps(physical_groups)
        return {
            "physical_groups": physical_groups,
            "physical_name_to_dimtag": physical_name_to_dimtag,
            "physical_dimtag_to_name": physical_dimtag_to_name,
            "boundary_line_tags": {
                "inlet": [inlet_line_tag],
                "outlet": [outlet_line_tag],
                "wall": wall_line_tags,
            },
        }

    def generate_mesh(self, gmsh_module):
        gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeMin", float(self.params["mesh_size_global"]))
        gmsh_module.option.setNumber("Mesh.MeshSizeMax", float(self.params["mesh_size_global"]))
        gmsh_module.model.mesh.generate(self.mesh_topology_dimension())

    def extract_mesh_data(self, gmsh_module):
        cell_data = extract_triangle_data(gmsh_module)
        boundary_data = extract_boundary_edges(gmsh_module, cell_data["node_index_map"])

        physical_groups = []
        for dim, tag in gmsh_module.model.getPhysicalGroups():
            name = gmsh_module.model.getPhysicalName(dim, tag)
            physical_groups.append((int(dim), int(tag), str(name)))

        physical_name_to_dimtag, physical_dimtag_to_name = build_dimtag_maps(physical_groups)
        mesh_data = {key: value for key, value in cell_data.items() if key != "node_index_map"}
        mesh_data.update(boundary_data)
        mesh_data["physical_groups"] = physical_groups
        mesh_data["physical_name_to_dimtag"] = physical_name_to_dimtag
        mesh_data["physical_dimtag_to_name"] = physical_dimtag_to_name
        return mesh_data

    def postprocess_mesh_data(self, mesh, mesh_data):
        boundary_edge = mesh_data.get("boundary_edge")
        boundary_marker = mesh_data.get("boundary_edge_marker")
        name_to_dimtag = mesh_data.get("physical_name_to_dimtag", {})

        if boundary_edge is not None and boundary_marker is not None:
            inlet_tag = name_to_dimtag.get("inlet", (1, -1))[1]
            outlet_tag = name_to_dimtag.get("outlet", (1, -1))[1]
            wall_tag = name_to_dimtag.get("wall", (1, -1))[1]

            mesh.boundary_edge = boundary_edge
            mesh.boundary_edge_marker = boundary_marker
            mesh.inlet = boundary_edge[boundary_marker == inlet_tag]
            mesh.outlet = boundary_edge[boundary_marker == outlet_tag]
            mesh.wall = boundary_edge[boundary_marker == wall_tag]
            mesh.design = mesh.wall
            mesh.fsi = mesh.wall

            mesh.inlet_nodes = np.unique(np.asarray(mesh.inlet, dtype=int).reshape(-1)) if mesh.inlet.size else np.asarray([], dtype=int)
            mesh.outlet_nodes = np.unique(np.asarray(mesh.outlet, dtype=int).reshape(-1)) if mesh.outlet.size else np.asarray([], dtype=int)
            mesh.wall_nodes = np.unique(np.asarray(mesh.wall, dtype=int).reshape(-1)) if mesh.wall.size else np.asarray([], dtype=int)
            mesh.transition_pipe_params = dict(self.params)
            mesh.fixed_x_left = float(self.params["fixed_x_left"])
            mesh.fixed_x_right = float(self.params["fixed_x_right"])

            fixed_segments = self.classify_fixed_segments(mesh)
            mesh.fixed_left_nodes = np.asarray(fixed_segments["fixed_left_nodes"], dtype=int)
            mesh.fixed_right_nodes = np.asarray(fixed_segments["fixed_right_nodes"], dtype=int)
            mesh.fixed_nodes = np.asarray(fixed_segments["fixed_nodes"], dtype=int)
            mesh.design_nodes = np.asarray(fixed_segments["design_nodes"], dtype=int)
            mesh.boundary_nodes = np.asarray(fixed_segments["boundary_nodes"], dtype=int)
            mesh.design_boundary_ids = mesh.design_nodes
            mesh.fixed_boundary_ids = mesh.fixed_nodes
            mesh.boundary_nodes_by_role = {
                "inlet": np.asarray(mesh.inlet_nodes, dtype=int),
                "outlet": np.asarray(mesh.outlet_nodes, dtype=int),
                "wall": np.asarray(mesh.wall_nodes, dtype=int),
                "fixed": np.asarray(mesh.fixed_nodes, dtype=int),
                "design": np.asarray(mesh.design_nodes, dtype=int),
                "boundary_cycle": np.asarray(mesh.boundary_nodes, dtype=int),
            }
            design_id_set = {int(node) for node in np.asarray(mesh.design_nodes, dtype=int).reshape(-1).tolist()}
            mesh.design_boundary_node_order = tuple(int(node) for node in np.asarray(mesh.boundary_nodes, dtype=int).reshape(-1).tolist() if int(node) in design_id_set)
            mesh.design_boundary_node_ids = tuple(int(node) for node in np.asarray(mesh.design_boundary_node_order, dtype=int).tolist())

        return mesh

    @staticmethod
    def classify_fixed_segments(
        mesh,
        fixed_x_left: float | None = None,
        fixed_x_right: float | None = None,
    ):
        coords = np.asarray(mesh.entity("node"), dtype=float)
        inlet_nodes = np.asarray(getattr(mesh, "inlet_nodes", []), dtype=int)
        outlet_nodes = np.asarray(getattr(mesh, "outlet_nodes", []), dtype=int)
        wall_nodes = np.asarray(getattr(mesh, "wall_nodes", []), dtype=int)
        boundary_nodes = np.asarray(
            np.unique(np.concatenate([inlet_nodes, outlet_nodes, wall_nodes])) if (inlet_nodes.size + outlet_nodes.size + wall_nodes.size) > 0 else np.asarray([], dtype=int),
            dtype=int,
        )

        if boundary_nodes.size == 0:
            empty = np.asarray([], dtype=int)
            return {
                "fixed_left_nodes": empty,
                "fixed_right_nodes": empty,
                "fixed_nodes": empty,
                "design_nodes": empty,
                "inlet_nodes": inlet_nodes,
                "outlet_nodes": outlet_nodes,
                "wall_nodes": wall_nodes,
                "boundary_nodes": boundary_nodes,
            }

        if fixed_x_left is None:
            fixed_x_left = getattr(mesh, "fixed_x_left", None)
        if fixed_x_right is None:
            fixed_x_right = getattr(mesh, "fixed_x_right", None)
        if fixed_x_left is None or fixed_x_right is None:
            params = getattr(mesh, "paper_pipe_params", {})
            if fixed_x_left is None:
                fixed_x_left = float(params.get("inlet_length", float(np.min(coords[:, 0]))))
            if fixed_x_right is None:
                fixed_x_right = float(params.get("inlet_length", float(np.min(coords[:, 0])))) + float(params.get("bend_length", 0.0))

        fixed_x_left = float(fixed_x_left)
        fixed_x_right = float(fixed_x_right)
        if fixed_x_left > fixed_x_right:
            raise ValueError("fixed_x_left must be less than or equal to fixed_x_right")

        tol = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])), 1.0) * 1.0e-10

        boundary_coords = coords[boundary_nodes]
        fixed_left_nodes = boundary_nodes[boundary_coords[:, 0] <= fixed_x_left + tol]
        fixed_right_nodes = boundary_nodes[boundary_coords[:, 0] >= fixed_x_right - tol]
        fixed_nodes = np.asarray(np.unique(np.concatenate([fixed_left_nodes, fixed_right_nodes])), dtype=int)
        design_nodes = np.asarray(np.setdiff1d(boundary_nodes, fixed_nodes, assume_unique=False), dtype=int)

        return {
            "fixed_left_nodes": np.asarray(fixed_left_nodes, dtype=int),
            "fixed_right_nodes": np.asarray(fixed_right_nodes, dtype=int),
            "fixed_nodes": np.asarray(fixed_nodes, dtype=int),
            "design_nodes": np.asarray(design_nodes, dtype=int),
            "inlet_nodes": inlet_nodes,
            "outlet_nodes": outlet_nodes,
            "wall_nodes": wall_nodes,
            "boundary_nodes": boundary_nodes,
            "fixed_x_left": fixed_x_left,
            "fixed_x_right": fixed_x_right,
        }

    def _classify_boundary_nodes(self, mesh, tol: float | None = None) -> dict[str, np.ndarray]:
        nodes = np.asarray(mesh.entity("node"), dtype=float)
        boundary_mask = self._boundary_edge_mask(mesh)
        boundary_edges = np.asarray(mesh.entity("edge"), dtype=int)[boundary_mask]
        boundary_ids = np.unique(boundary_edges.reshape(-1))
        if boundary_ids.size == 0:
            empty = np.asarray([], dtype=int)
            return {"inlet": empty, "outlet": empty, "wall": empty, "boundary_cycle": empty}
        if tol is None:
            span = np.ptp(nodes[boundary_ids], axis=0)
            tol = max(float(np.max(span)), 1.0) * 1.0e-10
        xmin = float(np.min(nodes[boundary_ids, 0]))
        xmax = float(np.max(nodes[boundary_ids, 0]))
        inlet = boundary_ids[np.abs(nodes[boundary_ids, 0] - xmin) <= tol]
        outlet = boundary_ids[np.abs(nodes[boundary_ids, 0] - xmax) <= tol]
        wall = np.setdiff1d(boundary_ids, np.unique(np.concatenate([inlet, outlet])), assume_unique=False)
        boundary_cycle = self._order_boundary_loop_node_ids(
            mesh,
            boundary_ids,
            fallback_center=tuple(nodes[boundary_ids].mean(axis=0).tolist()),
        )
        return {
            "inlet": np.asarray(np.unique(inlet), dtype=int),
            "outlet": np.asarray(np.unique(outlet), dtype=int),
            "wall": np.asarray(np.unique(wall), dtype=int),
            "boundary_cycle": np.asarray(boundary_cycle, dtype=int),
        }

    def build_mesh_from_boundary_points(self, boundary_points) -> TriangleMesh:
        coords = np.asarray(boundary_points, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
            raise ValueError("boundary_points must be a polygon with at least three vertices")
        gmsh_module = self.gmsh
        owns_session = not gmsh_module.isInitialized()
        if owns_session:
            gmsh_module.initialize()
        try:
            gmsh_module.model.add(self.model_name())
            occ = gmsh_module.model.occ
            points = [occ.addPoint(float(x), float(y), 0.0) for x, y in coords]
            lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
            loop = occ.addCurveLoop(lines)
            occ.addPlaneSurface([loop])
            occ.synchronize()
            normalized = str(self.params["mesh_size_profile"]).casefold()
            if normalized in {"cashocs", "default", "uniform"}:
                gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh_module.option.setNumber("Mesh.CharacteristicLengthMin", float(self.params["mesh_size_global"]))
                gmsh_module.option.setNumber("Mesh.CharacteristicLengthMax", float(self.params["mesh_size_global"]))
            elif normalized == "graded":
                mesh_size_inner = float(self.params["mesh_size_inner"] if self.params["mesh_size_inner"] is not None else max(0.25 * float(self.params["mesh_size_global"]), 0.03))
                mesh_size_outer = float(self.params["mesh_size_outer"] if self.params["mesh_size_outer"] is not None else float(self.params["mesh_size_global"]))
                mesh_size_transition = float(
                    self.params["mesh_size_transition"]
                    if self.params["mesh_size_transition"] is not None
                    else max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 2.0 * mesh_size_outer)
                )
                if mesh_size_inner <= 0.0 or mesh_size_outer <= 0.0 or mesh_size_transition <= 0.0:
                    raise ValueError("mesh size parameters must be positive")
                if mesh_size_inner > mesh_size_outer:
                    raise ValueError("mesh_size_inner must be less than or equal to mesh_size_outer")
                gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh_module.option.setNumber("Mesh.MeshSizeMin", mesh_size_inner)
                gmsh_module.option.setNumber("Mesh.MeshSizeMax", mesh_size_outer)
                gmsh_module.option.setNumber("Mesh.Algorithm", 5)
                distance_field = gmsh_module.model.mesh.field.add("Distance")
                gmsh_module.model.mesh.field.setNumbers(distance_field, "CurvesList", lines)
                gmsh_module.model.mesh.field.setNumber(distance_field, "Sampling", 100)
                threshold_field = gmsh_module.model.mesh.field.add("Threshold")
                gmsh_module.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
                gmsh_module.model.mesh.field.setNumber(threshold_field, "SizeMin", mesh_size_inner)
                gmsh_module.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size_outer)
                gmsh_module.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
                gmsh_module.model.mesh.field.setNumber(threshold_field, "DistMax", mesh_size_transition)
                gmsh_module.model.mesh.field.setAsBackgroundMesh(threshold_field)
            else:
                raise ValueError(f"Unsupported mesh_size_profile: {self.params['mesh_size_profile']!r}")
            gmsh_module.model.mesh.generate(2)
            node_tags, node_coords, _ = gmsh_module.model.mesh.getNodes()
            node = bm.from_numpy(node_coords.reshape(-1, 3)[:, 0:2])
            tri_type = gmsh_module.model.mesh.getElementType("triangle", 1)
            types, elemTags, elemNodeTags = gmsh_module.model.mesh.getElements(2)
            cell = None
            for etype, element_nodes in zip(types, elemNodeTags):
                if etype == tri_type:
                    nn = gmsh_module.model.mesh.getElementProperties(etype)[3]
                    cell = bm.array(element_nodes, dtype=bm.int32).reshape(-1, nn) - 1
                    break
            if cell is None:
                raise RuntimeError("polygon mesh did not produce triangle cells")
            nn = len(node)
            valid = bm.zeros(nn, dtype=bm.bool)
            valid = bm.set_at(valid, cell, True)
            node = node[valid]
            idx_map = bm.zeros(nn, dtype=cell.dtype)
            idx_map = bm.set_at(idx_map, valid, bm.arange(valid.sum(), dtype=bm.int64))
            cell = idx_map[cell]
            mesh = TriangleMesh(node, cell)
            roles = self._classify_boundary_nodes(mesh)
            mesh.inlet_nodes = np.asarray(roles["inlet"], dtype=int)
            mesh.outlet_nodes = np.asarray(roles["outlet"], dtype=int)
            mesh.wall_nodes = np.asarray(roles["wall"], dtype=int)
            mesh.fixed_x_left = float(self.params["fixed_x_left"])
            mesh.fixed_x_right = float(self.params["fixed_x_right"])
            fixed_segments = self.classify_fixed_segments(
                mesh,
                fixed_x_left=mesh.fixed_x_left,
                fixed_x_right=mesh.fixed_x_right,
            )
            mesh.fixed_left_nodes = np.asarray(fixed_segments["fixed_left_nodes"], dtype=int)
            mesh.fixed_right_nodes = np.asarray(fixed_segments["fixed_right_nodes"], dtype=int)
            mesh.fixed_nodes = np.asarray(fixed_segments["fixed_nodes"], dtype=int)
            mesh.design_nodes = np.asarray(fixed_segments["design_nodes"], dtype=int)
            boundary_cycle = np.asarray(roles["boundary_cycle"], dtype=int)
            mesh.boundary_nodes = boundary_cycle
            mesh.design_boundary_ids = mesh.design_nodes
            mesh.fixed_boundary_ids = mesh.fixed_nodes
            design_id_set = {int(node_id) for node_id in np.asarray(mesh.design_nodes, dtype=int).reshape(-1).tolist()}
            mesh.design_boundary_node_order = tuple(int(node_id) for node_id in boundary_cycle.tolist() if int(node_id) in design_id_set)
            mesh.design_boundary_node_ids = tuple(int(node_id) for node_id in mesh.design_boundary_node_order)
            mesh.boundary_nodes_by_role = {
                "inlet": np.asarray(mesh.inlet_nodes, dtype=int),
                "outlet": np.asarray(mesh.outlet_nodes, dtype=int),
                "wall": np.asarray(mesh.wall_nodes, dtype=int),
                "fixed": np.asarray(mesh.fixed_nodes, dtype=int),
                "design": np.asarray(mesh.design_nodes, dtype=int),
                "boundary_cycle": np.asarray(mesh.boundary_nodes, dtype=int),
            }
            return mesh
        finally:
            if owns_session and gmsh_module.isInitialized():
                gmsh_module.finalize()

    def run(self, visualize: bool = False, write_path=None):
        gmsh_module = self.gmsh
        owns_session = not gmsh_module.isInitialized()
        if owns_session:
            gmsh_module.initialize()

        try:
            gmsh_module.model.add(self.model_name())
            self._build_geometry(gmsh_module, self.params)
            self.generate_mesh(gmsh_module)

            mesh_data = self.extract_mesh_data(gmsh_module)
            if write_path is not None:
                gmsh_module.write(str(Path(write_path)))
            if visualize:
                gmsh_module.fltk.run()

            self._mesh_data_cache = mesh_data
            return mesh_data
        finally:
            if owns_session and gmsh_module.isInitialized():
                gmsh_module.finalize()

    def mesh_data(self, visualize: bool = False, write_path=None):
        if self._mesh_data_cache is None:
            return self.run(visualize=visualize, write_path=write_path)
        return self._mesh_data_cache

    def init_mesh(self, visualize: bool = False, write_path=None):
        mesh_data = self.run(visualize=visualize, write_path=write_path)
        mesh = TriangleMesh(mesh_data["node"], mesh_data["triangle"])
        if "triangle_region" in mesh_data:
            mesh.celldata["region"] = mesh_data["triangle_region"]
        return self.postprocess_mesh_data(mesh, mesh_data)
