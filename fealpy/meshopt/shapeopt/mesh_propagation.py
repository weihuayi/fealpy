"""网格传播器骨架。

这层负责把边界位移传播到内部点，尽量接近 cashocs 的变形流程。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Any, Mapping

from fealpy.backend import bm
from fealpy.fem import BilinearForm, DirichletBC, LinearElasticityIntegrator
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import csr_matrix
from fealpy.material import LinearElasticMaterial
from fealpy.solver import spsolve as fealpy_spsolve

from .benchmark_common import get_value
from scipy.sparse.csgraph import dijkstra


@dataclass(slots=True)
class TrialState:
    """试探状态。"""

    trial_mesh: Any
    trial_design_boundary: Any
    trial_boundary_displacement: Any = None
    trial_cache: Any = None
    trial_objective: float | None = None
    quality_info: Any = None
    extension_path: str | None = None
    used_deformation_extension: bool = False
    restart_optimization: bool = False
    deformation_extension_result: Any = None


@dataclass(slots=True)
class RemeshResult:
    """重网格审计结果。"""

    remeshed_state: Any
    remeshed: bool
    requires_remesh: bool
    restart_optimization: bool = False
    quality_info: Any = None
    handler_name: str | None = None
    remesh_output: Any = None
    reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self) -> list[tuple[str, Any]]:
        return [
            ("remeshed_state", self.remeshed_state),
            ("remeshed", self.remeshed),
            ("requires_remesh", self.requires_remesh),
            ("restart_optimization", self.restart_optimization),
            ("quality_info", self.quality_info),
            ("handler_name", self.handler_name),
            ("remesh_output", self.remesh_output),
            ("reason", self.reason),
        ]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.items())


def _handler_name(handler: Any) -> str | None:
    """给可调用重网格处理器生成一个可审计名称。"""
    if handler is None:
        return None
    name = getattr(handler, "__name__", None)
    if name:
        return name
    return type(handler).__name__


def _is_numeric(value: Any) -> bool:
    """判断值是否是单个标量。"""
    return isinstance(value, Real)


def _add_value(left: Any, right: Any) -> Any:
    """把标量、向量或映射做加法。"""
    if _is_numeric(left) and _is_numeric(right):
        return float(left) + float(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        return {
            key: _add_value(left.get(key, 0.0), right.get(key, 0.0))
            for key in keys
        }
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return type(left)(_add_value(lhs, rhs) for lhs, rhs in zip(left, right))
    return right if right is not None else left


def _copy_mesh(mesh: Any) -> Any:
    """复制网格对象。"""
    if isinstance(mesh, Mapping):
        return dict(mesh)
    if hasattr(mesh, "copy"):
        return mesh.copy()
    return deepcopy(mesh)


def _set_mesh_nodes(mesh: Any, coordinates: Any) -> Any:
    """写回节点坐标。"""
    coordinates = bm.asarray(coordinates, dtype=float)
    if isinstance(mesh, Mapping):
        updated_mesh = dict(mesh)
        if "node" in updated_mesh:
            updated_mesh["node"] = coordinates
        elif "nodes" in updated_mesh:
            updated_mesh["nodes"] = coordinates
        else:
            updated_mesh["node"] = coordinates
        return updated_mesh
    if hasattr(mesh, "node"):
        mesh.node[...] = coordinates
        return mesh
    if hasattr(mesh, "nodes"):
        mesh.nodes[...] = coordinates
        return mesh
    return mesh


def get_boundary_nodes(mesh: Any, propagation_parameters: Any) -> Any | None:
    nodes = get_value(propagation_parameters, "boundary_nodes", "design_boundary_nodes")
    if nodes is None and hasattr(mesh, "boundary_node_index") and callable(mesh.boundary_node_index):
        nodes = mesh.boundary_node_index()
    return None if nodes is None else bm.asarray(nodes, dtype=int)


def get_fixed_nodes(propagation_parameters: Any) -> Any:
    nodes = get_value(propagation_parameters, "fixed_nodes")
    return bm.asarray([], dtype=int) if nodes is None else bm.asarray(nodes, dtype=int)


def _as_coordinate_vector(value: Any, dim: int) -> Any:
    """把位移转成坐标向量。"""
    arr = bm.asarray(value, dtype=float)
    if arr.ndim == 0:
        return bm.full(dim, float(arr), dtype=float)
    if arr.size == dim:
        return arr.reshape(dim).astype(float, copy=False)
    if arr.size == 1:
        return bm.full(dim, float(arr.reshape(())), dtype=float)
    raise ValueError("boundary displacement dimension mismatch")


def _mesh_edges(cells: Any) -> Any:
    """从单元连接中提取去重边。"""
    cells = bm.asarray(cells, dtype=int)
    if cells.ndim != 2 or cells.shape[1] < 2:
        return bm.zeros((0, 2), dtype=int)
    pairs = bm.asarray(tuple(combinations(range(cells.shape[1]), 2)), dtype=int)
    edges = cells[:, pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return bm.unique(edges, axis=0)


def _build_graph_laplacian(num_nodes: int, cells: Any) -> Any:
    """根据单元连接构造图拉普拉斯矩阵。"""
    edges = _mesh_edges(cells)
    if edges.size == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=float)

    degree = bm.index_add(bm.zeros(num_nodes, dtype=float), edges.ravel(), 1.0)
    off_rows = bm.concatenate((edges[:, 0], edges[:, 1]))
    off_cols = bm.concatenate((edges[:, 1], edges[:, 0]))
    rows = bm.concatenate((off_rows, bm.arange(num_nodes, dtype=int)))
    cols = bm.concatenate((off_cols, bm.arange(num_nodes, dtype=int)))
    data = bm.concatenate((bm.full(off_rows.shape[0], -1.0, dtype=float), degree))
    return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def _update_boundary_coordinates(
    current_mesh: Any,
    boundary_displacement: Any,
    geometry_contract: Any,
) -> Any:
    """只更新字典式网格里的设计边界坐标。"""
    design_boundary_ids = set(getattr(geometry_contract, "design_boundary_ids", ()) or ())
    fixed_boundary_ids = set(getattr(geometry_contract, "fixed_boundary_ids", ()) or ())
    boundary_coordinates = get_value(current_mesh, "boundary_coordinates")
    if boundary_coordinates is None or not isinstance(boundary_coordinates, Mapping):
        return current_mesh

    displacement_map = boundary_displacement if isinstance(boundary_displacement, Mapping) else {boundary_id: boundary_displacement for boundary_id in design_boundary_ids}
    updated_boundary_coordinates = dict(boundary_coordinates)
    mutable_ids = [boundary_id for boundary_id in design_boundary_ids if boundary_id not in fixed_boundary_ids and boundary_id in updated_boundary_coordinates and displacement_map.get(boundary_id) is not None]
    if mutable_ids:
        current_values = bm.asarray([updated_boundary_coordinates[boundary_id] for boundary_id in mutable_ids], dtype=float)
        displacement_values = bm.asarray([displacement_map[boundary_id] for boundary_id in mutable_ids], dtype=float)
        updated_boundary_coordinates.update({boundary_id: tuple(current_value + displacement_value) for boundary_id, current_value, displacement_value in zip(mutable_ids, current_values, displacement_values)})

    if isinstance(current_mesh, Mapping):
        updated_mesh = dict(current_mesh)
        updated_mesh["boundary_coordinates"] = updated_boundary_coordinates
        return updated_mesh

    if hasattr(current_mesh, "boundary_coordinates"):
        current_mesh.boundary_coordinates = updated_boundary_coordinates
    return current_mesh


def _shortest_distances_to_sources(
    mesh: Any,
    source_nodes: Any,
) -> Any | None:
    """计算每个节点到给定源节点集合的近似图距离。"""
    nodes = mesh.node
    cells = mesh.cell
    if nodes is None or cells is None:
        return None
    if source_nodes.size == 0:
        return None

    valid_sources = bm.unique(source_nodes[(source_nodes >= 0) & (source_nodes < nodes.shape[0])].astype(int, copy=False))
    if valid_sources.size == 0:
        return None

    edges = _mesh_edges(cells)
    if edges.size == 0:
        return bm.full(nodes.shape[0], bm.inf, dtype=float)

    lengths = bm.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    lengths[lengths <= 0.0] = 1.0
    rows = bm.concatenate((edges[:, 0], edges[:, 1]))
    cols = bm.concatenate((edges[:, 1], edges[:, 0]))
    data = bm.concatenate((lengths, lengths))
    graph = csr_matrix((data, (rows, cols)), shape=(nodes.shape[0], nodes.shape[0]))
    distances = bm.asarray(dijkstra(graph.to_scipy(), directed=False, indices=bm.to_numpy(valid_sources)), dtype=float)
    if distances.ndim == 1:
        return distances
    return bm.min(distances, axis=0)


def _smoothstep(value: Any) -> Any:
    """简单的 C1 平滑插值。"""
    return value * value * (3.0 - 2.0 * value)


def _compute_node_stiffness_profile(
    mesh: Any,
    propagation_parameters: Any,
) -> Any | None:
    """计算节点刚度分布。"""
    nodes = mesh.node
    if nodes is None:
        return None

    boundary_nodes = get_boundary_nodes(mesh, propagation_parameters)
    if boundary_nodes is None:
        return bm.ones(nodes.shape[0], dtype=float)

    fixed_nodes = get_fixed_nodes(propagation_parameters)
    design_nodes = get_value(propagation_parameters, "design_boundary_nodes", "shape_bdry_def", "deformable_boundary_nodes")
    design_nodes = boundary_nodes if design_nodes is None else bm.asarray(design_nodes, dtype=int)
    constrained_nodes = bm.unique(bm.concatenate((design_nodes, fixed_nodes)))

    use_distance_mu = bool(get_value(propagation_parameters, "use_distance_mu", default=False))
    if use_distance_mu:
        source_boundaries = get_value(propagation_parameters, "boundaries_dist")
        source_nodes = boundary_nodes if source_boundaries is None or len(source_boundaries) == 0 else bm.asarray(source_boundaries, dtype=int)
        distances = _shortest_distances_to_sources(mesh, source_nodes)
        if distances is None:
            return bm.ones(nodes.shape[0], dtype=float)

        dist_min = float(get_value(propagation_parameters, "dist_min", default=1.0))
        dist_max = float(get_value(propagation_parameters, "dist_max", default=1.0))
        mu_min = float(get_value(propagation_parameters, "mu_min", default=1.0))
        mu_max = float(get_value(propagation_parameters, "mu_max", default=1.0))
        smooth_mu = bool(get_value(propagation_parameters, "smooth_mu", default=False))

        profile = bm.full(nodes.shape[0], mu_max, dtype=float)
        if abs(dist_max - dist_min) <= 1e-12:
            profile[:] = mu_max
            profile[distances <= dist_min] = mu_min
            return profile

        span = bm.maximum(dist_max - dist_min, 1e-12)
        scaled = bm.clip((distances - dist_min) / span, 0.0, 1.0)
        if smooth_mu:
            scaled = _smoothstep(scaled)
        profile = mu_min + (mu_max - mu_min) * scaled
        profile[distances <= dist_min] = mu_min
        profile[distances >= dist_max] = mu_max
        return profile

    mu_def = float(get_value(propagation_parameters, "mu_def", default=1.0))
    mu_fix = float(get_value(propagation_parameters, "mu_fix", default=1.0))
    if abs(mu_def - mu_fix) <= 1e-12:
        return bm.full(nodes.shape[0], mu_fix, dtype=float)

    boundary_values = bm.full(nodes.shape[0], mu_fix, dtype=float)
    boundary_values[design_nodes] = mu_def
    boundary_values[fixed_nodes] = mu_fix

    profile = boundary_values.copy()
    interior_nodes = bm.setdiff1d(bm.arange(nodes.shape[0], dtype=int), constrained_nodes, assume_unique=False)
    if interior_nodes.size == 0:
        return profile

    cells = mesh.cell
    if cells is None:
        return profile
    laplacian = _build_graph_laplacian(nodes.shape[0], cells)
    constrained_nodes = constrained_nodes.astype(int, copy=False)
    lap_ii = laplacian[interior_nodes][:, interior_nodes]
    lap_ib = laplacian[interior_nodes][:, constrained_nodes]
    rhs = -(lap_ib @ boundary_values[constrained_nodes])
    interior_solution = fealpy_spsolve(lap_ii, rhs, solver=get_value(propagation_parameters, "linear_solver", default="scipy"))
    profile[interior_nodes] = interior_solution
    return profile


def _build_weighted_graph_laplacian(num_nodes: int, cells: Any, node_weights: Any) -> Any:
    """根据节点刚度构造加权图拉普拉斯。"""
    edges = _mesh_edges(cells)
    if edges.size == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=float)

    eps = 1e-12
    node_weights = bm.asarray(node_weights, dtype=float)
    conductance = 2.0 / bm.maximum(node_weights[edges[:, 0]] + node_weights[edges[:, 1]], eps)
    degree = bm.bincount(edges[:, 0], weights=conductance, minlength=num_nodes).astype(float)
    degree += bm.bincount(edges[:, 1], weights=conductance, minlength=num_nodes).astype(float)
    rows = bm.concatenate((edges[:, 0], edges[:, 1], bm.arange(num_nodes, dtype=int)))
    cols = bm.concatenate((edges[:, 1], edges[:, 0], bm.arange(num_nodes, dtype=int)))
    data = bm.concatenate((-conductance, -conductance, degree))
    return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def _solve_weighted_extension(
    mesh: Any,
    boundary_displacement: Any,
    propagation_parameters: Any,
    node_weights: Any | None = None,
) -> Any | None:
    """使用加权调和扩展求解全局位移场。"""
    nodes = mesh.node
    cells = mesh.cell
    if nodes is None or cells is None:
        return None

    boundary_nodes = get_boundary_nodes(mesh, propagation_parameters)
    if boundary_nodes is None or boundary_nodes.size == 0:
        return None

    fixed_nodes = get_fixed_nodes(propagation_parameters)
    constrained_nodes = bm.unique(bm.concatenate((boundary_nodes, fixed_nodes)))
    dim = nodes.shape[1]
    updated_displacement = bm.zeros_like(nodes, dtype=float)
    if isinstance(boundary_displacement, Mapping):
        if boundary_displacement:
            keys = bm.asarray(list(boundary_displacement.keys()), dtype=int)
            values = bm.asarray([_as_coordinate_vector(value, dim) for value in boundary_displacement.values()], dtype=float)
            valid = (keys >= 0) & (keys < updated_displacement.shape[0])
            updated_displacement[keys[valid]] = values[valid]
    else:
        array = bm.asarray(boundary_displacement, dtype=float)
        if array.ndim == 2 and array.shape[0] == updated_displacement.shape[0] and array.shape[1] >= dim:
            updated_displacement[:] = array[:, :dim]
        else:
            updated_displacement[boundary_nodes] = _as_coordinate_vector(array, dim)
    updated_displacement[fixed_nodes] = 0.0

    if node_weights is None:
        node_weights = bm.ones(nodes.shape[0], dtype=float)
    node_weights = bm.asarray(node_weights, dtype=float)

    constrained_nodes_sorted = constrained_nodes.astype(int, copy=False)
    interior_nodes = bm.setdiff1d(bm.arange(nodes.shape[0], dtype=int), constrained_nodes_sorted, assume_unique=False)
    if interior_nodes.size == 0:
        return updated_displacement

    laplacian = _build_weighted_graph_laplacian(nodes.shape[0], cells, node_weights)
    lap_ii = laplacian[interior_nodes][:, interior_nodes]
    lap_ib = laplacian[interior_nodes][:, constrained_nodes_sorted]

    constrained_displacement = updated_displacement[constrained_nodes_sorted]
    rhs = -(lap_ib @ constrained_displacement)
    interior_solution = fealpy_spsolve(lap_ii, rhs, solver=get_value(propagation_parameters, "linear_solver", default="scipy"))
    updated_displacement[interior_nodes] = bm.asarray(interior_solution, dtype=float)

    return updated_displacement


def _compute_deformation_metrics(
    mesh: Any,
    deformation: Any,
) -> tuple[Any, Any] | tuple[None, None]:
    """计算每个单元的体积变化与角度变化指标。"""
    nodes = mesh.node
    if isinstance(mesh, Mapping):
        cells = mesh.get("cell", mesh.get("cells"))
    else:
        cells = get_value(mesh, "cell", "cells")
    if cells is not None:
        cells = bm.asarray(cells, dtype=int)
    if nodes is None or cells is None or nodes.shape[1] != 2:
        return None, None

    deformation = bm.asarray(deformation, dtype=float)
    triangles = bm.asarray(cells, dtype=int)
    cell_nodes = nodes[triangles]
    cell_deformation = deformation[triangles]
    jacobian = bm.stack((cell_nodes[:, 1] - cell_nodes[:, 0], cell_nodes[:, 2] - cell_nodes[:, 0]), axis=-1)
    determinants = bm.linalg.det(jacobian)
    if bm.any(determinants == 0.0):
        return bm.full(triangles.shape[0], 0.0, dtype=float), bm.full(triangles.shape[0], bm.inf, dtype=float)
    displacement_jacobian = bm.stack((cell_deformation[:, 1] - cell_deformation[:, 0], cell_deformation[:, 2] - cell_deformation[:, 0]), axis=-1)
    grad_u = displacement_jacobian @ bm.linalg.inv(jacobian)
    deformation_gradient = bm.eye(2, dtype=float)[None, :, :] + grad_u
    return bm.linalg.det(deformation_gradient), bm.linalg.norm(grad_u, ord="fro", axis=(1, 2))


def _check_deformation_limits(
    mesh: Any,
    deformation: Any,
    propagation_parameters: Any,
) -> dict[str, Any]:
    """检查 a priori 变形限制。"""
    deformation = bm.asarray(deformation, dtype=float)
    volume_change_limit = float(get_value(propagation_parameters, "volume_change", default=float("inf")))
    angle_change_limit = float(get_value(propagation_parameters, "angle_change", default=float("inf")))
    volume_change, angle_change = _compute_deformation_metrics(mesh, deformation)
    if volume_change is None or angle_change is None:
        return {
            "accepted": True,
            "volume_change_limit": volume_change_limit,
            "angle_change_limit": angle_change_limit,
            "volume_change": None,
            "angle_change": None,
        }

    volume_ok = True
    if not bm.isinf(volume_change_limit):
        lower = 1.0 / max(volume_change_limit, 1e-12)
        upper = volume_change_limit
        volume_ok = bool(bm.all((volume_change >= lower) & (volume_change <= upper)))

    angle_ok = True
    if not bm.isinf(angle_change_limit):
        angle_ok = bool(bm.all(angle_change <= angle_change_limit))

    return {
        "accepted": bool(volume_ok and angle_ok),
        "volume_change_limit": volume_change_limit,
        "angle_change_limit": angle_change_limit,
        "volume_change": volume_change,
        "angle_change": angle_change,
        "volume_ok": volume_ok,
        "angle_ok": angle_ok,
    }


def _triangle_cell_angles(triangle_nodes: Any) -> Any | None:
    """计算二维三角形单元的三个内角。"""
    points = bm.asarray(triangle_nodes, dtype=float)
    if points.shape[0] != 3 or points.shape[1] < 2:
        return None

    lhs = bm.stack((points[1] - points[0], points[0] - points[1], points[0] - points[2]), axis=0)
    rhs = bm.stack((points[2] - points[0], points[2] - points[1], points[1] - points[2]), axis=0)
    lhs_norm = bm.linalg.norm(lhs, axis=1)
    rhs_norm = bm.linalg.norm(rhs, axis=1)
    if bm.any(lhs_norm <= 0.0) or bm.any(rhs_norm <= 0.0):
        return None
    cosine = bm.sum(lhs * rhs, axis=1) / (lhs_norm * rhs_norm)
    return bm.arccos(bm.clip(cosine, -1.0, 1.0))


def _triangle_quality_from_measure(
    triangle_nodes: Any,
    quality_measure: str,
) -> float:
    """根据 cashocs 风格的度量计算单个二维三角形的质量。"""
    points = bm.asarray(triangle_nodes, dtype=float)
    if points.shape[0] != 3 or points.shape[1] < 2:
        return 0.0

    area = 0.5 * abs(
        (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
        - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
    )
    if area <= 0.0:
        return 0.0

    quality_measure = str(quality_measure)
    if quality_measure in {"skewness", "maximum_angle"}:
        angles = _triangle_cell_angles(points)
        if angles is None:
            return 0.0
        opt_angle = bm.pi / 3.0
        if quality_measure == "skewness":
            angle_quality = 1.0 - bm.maximum((angles - opt_angle) / (bm.pi - opt_angle), (opt_angle - angles) / opt_angle)
            return float(bm.clip(bm.min(angle_quality), 0.0, 1.0))
        max_angle = float(bm.max(angles))
        return float(bm.clip(1.0 - max((max_angle - opt_angle) / (bm.pi - opt_angle), 0.0), 0.0, 1.0))

    side_lengths = bm.asarray(
        [
            float(bm.linalg.norm(points[1] - points[0])),
            float(bm.linalg.norm(points[2] - points[1])),
            float(bm.linalg.norm(points[0] - points[2])),
        ],
        dtype=float,
    )
    if bm.any(side_lengths <= 0.0):
        return 0.0

    if quality_measure == "radius_ratios":
        perimeter = float(bm.sum(side_lengths))
        circumradius_factor = float(bm.prod(side_lengths))
        if perimeter <= 0.0 or circumradius_factor <= 0.0:
            return 0.0
        inradius = 2.0 * area / perimeter
        circumradius = circumradius_factor / (4.0 * area)
        if circumradius <= 0.0:
            return 0.0
        return float(bm.clip(2.0 * inradius / circumradius, 0.0, 1.0))

    if quality_measure == "condition_number":
        jacobian = bm.stack((points[1] - points[0], points[2] - points[0]), axis=-1)
        determinant = float(bm.linalg.det(jacobian))
        if not bm.isfinite(determinant) or abs(determinant) <= 0.0:
            return 0.0
        inverse = bm.linalg.inv(jacobian)
        frobenius = float(bm.linalg.norm(jacobian, ord="fro") * bm.linalg.norm(inverse, ord="fro"))
        if not bm.isfinite(frobenius) or frobenius <= 0.0:
            return 0.0
        return float(bm.clip(bm.sqrt(2.0) / frobenius, 0.0, 1.0))

    raise ValueError(f"Unsupported quality_measure: {quality_measure}")


def _compute_cell_quality_values(mesh: Any, quality_measure: str) -> Any | None:
    """计算每个单元的质量值。"""
    nodes = mesh.node
    cells = mesh.cell
    if nodes is None or cells is None or cells.shape[1] != 3 or nodes.shape[1] < 2:
        return None

    triangles = bm.asarray(cells, dtype=int)
    return bm.asarray([_triangle_quality_from_measure(nodes[triangle], quality_measure) for triangle in triangles], dtype=float)


def _aggregate_cell_quality_values(
    cell_qualities: Any,
    quality_type: str,
    quantile: float,
) -> float:
    """将单元质量聚合成一个标量质量。"""
    values = bm.asarray(cell_qualities, dtype=float)
    if values.size == 0:
        return 0.0

    quality_type = str(quality_type)
    if quality_type == "min":
        return float(bm.min(values))
    if quality_type == "avg":
        return float(bm.mean(values))
    if quality_type == "quantile":
        sorted_values = bm.sort(values)
        if sorted_values.size == 1:
            return float(sorted_values[0])
        q = min(max(float(quantile), 0.0), 1.0)
        position = q * (sorted_values.size - 1)
        left = int(bm.floor(position))
        right = int(bm.ceil(position))
        if left == right:
            return float(sorted_values[left])
        weight = position - left
        return float((1.0 - weight) * sorted_values[left] + weight * sorted_values[right])
    raise ValueError(f"Unsupported quality_type: {quality_type}")


def compute_mesh_quality(
    mesh: Any,
    quality_type: str = "min",
    quality_measure: str = "skewness",
    quantile: float = 0.0,
) -> float:
    """按 cashocs 风格计算网格质量。"""
    cell_qualities = _compute_cell_quality_values(mesh, quality_measure)
    if cell_qualities is not None:
        return _aggregate_cell_quality_values(cell_qualities, quality_type, quantile)

    if isinstance(mesh, Mapping) and "quality" in mesh:
        value = mesh["quality"]
        if isinstance(value, Mapping):
            value = value.get("quality", value.get("value", 0.0))
        return float(value)

    if hasattr(mesh, "quality"):
        quality = getattr(mesh, "quality")
        if callable(quality):
            quality = quality()
        if isinstance(quality, Mapping):
            quality = quality.get("quality", quality.get("value", 0.0))
        return float(quality)

    return 0.0


def _classify_mesh_quality(
    quality: float,
    tol_lower: float,
    tol_upper: float,
    *,
    has_negative_cells: bool = False,
    test_for_intersections: bool = True,
) -> dict[str, Any]:
    """把质量值分成 good / marginal / poor 三档。"""
    if has_negative_cells and test_for_intersections:
        return {
            "status": "poor",
            "quality_state": "poor",
            "accepted": False,
            "needs_remesh": False,
            "reason": "negative_cells",
        }

    if quality < tol_lower:
        return {
            "status": "rejected",
            "quality_state": "poor",
            "accepted": False,
            "needs_remesh": False,
            "reason": "quality_below_lower_tolerance",
        }

    if quality < tol_upper:
        return {
            "status": "marginal",
            "quality_state": "marginal",
            "accepted": True,
            "needs_remesh": False,
            "reason": "quality_between_tolerances",
        }

    return {
        "status": "accepted",
        "quality_state": "good",
        "accepted": True,
        "needs_remesh": False,
        "reason": None,
    }


def _mesh_quality_parameters(propagation_parameters: Any) -> tuple[str, str, float, float, float]:
    """读取质量参数。"""
    quality_measure = str(
        get_value(
            propagation_parameters,
            "mesh_quality_measure",
            "measure",
            default="skewness",
        )
    )
    quality_type = str(
        get_value(
            propagation_parameters,
            "mesh_quality_type",
            "type",
            default="min",
        )
    )
    quality_quantile = float(
        get_value(
            propagation_parameters,
            "quality_quantile",
            "quantile",
            default=0.0,
        )
    )
    tol_lower = float(
        get_value(
            propagation_parameters,
            "mesh_quality_tol_lower",
            "tol_lower",
            default=0.0,
        )
    )
    tol_upper = float(
        get_value(
            propagation_parameters,
            "mesh_quality_tol_upper",
            "tol_upper",
            default=1e-15,
        )
    )
    return quality_measure, quality_type, quality_quantile, tol_lower, tol_upper


def _boundary_normals_from_centroid(mesh: Any, boundary_nodes: Any) -> Any | None:
    """用几何中心近似边界法向。"""
    nodes = mesh.node
    cells = mesh.cell
    if nodes is None or cells is None:
        return None
    boundary_nodes = bm.asarray(boundary_nodes, dtype=int)
    if nodes.shape[1] != 2:
        centroid = nodes.mean(axis=0)
        vectors = nodes[boundary_nodes] - centroid
        norms = bm.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
        return vectors / norms[:, None]

    edges = _mesh_edges(cells)
    if edges.size == 0:
        return bm.zeros((boundary_nodes.size, 2), dtype=float)
    pairs = bm.asarray(tuple(combinations(range(bm.asarray(cells, dtype=int).shape[1]), 2)), dtype=int)
    all_edges = bm.asarray(cells, dtype=int)[:, pairs].reshape(-1, 2)
    all_edges.sort(axis=1)
    unique_edges, counts = bm.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    centroid = nodes.mean(axis=0)
    p0 = nodes[boundary_edges[:, 0]]
    p1 = nodes[boundary_edges[:, 1]]
    tangent = p1 - p0
    candidate = bm.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    midpoint = 0.5 * (p0 + p1)
    flip = bm.einsum("ij,ij->i", candidate, midpoint - centroid) < 0.0
    candidate[flip] *= -1.0
    norms = bm.linalg.norm(candidate, axis=1)
    candidate = bm.divide(candidate, norms[:, None], out=bm.zeros_like(candidate), where=norms[:, None] > 0.0)
    normals = bm.zeros((nodes.shape[0], 2), dtype=float)
    normals = bm.index_add(normals, boundary_edges[:, 0], candidate)
    normals = bm.index_add(normals, boundary_edges[:, 1], candidate)
    values = normals[boundary_nodes]
    norms = bm.linalg.norm(values, axis=1)
    fallback = nodes[boundary_nodes] - centroid
    fallback_norms = bm.linalg.norm(fallback, axis=1)
    fallback = bm.divide(fallback, fallback_norms[:, None], out=bm.zeros_like(fallback), where=fallback_norms[:, None] > 0.0)
    fallback[fallback_norms == 0.0] = bm.asarray([1.0, 0.0], dtype=float)
    values = bm.divide(values, norms[:, None], out=bm.zeros_like(values), where=norms[:, None] > 0.0)
    zero_mask = norms == 0.0
    if bm.any(zero_mask):
        values[zero_mask] = fallback[zero_mask]
    return values


def _solve_linear_elasticity_extension(
    mesh: Any,
    boundary_displacement: Any,
    propagation_parameters: Any,
) -> Any | None:
    """用线弹性方程把边界位移扩展为全局位移场。"""
    use_distance_mu = bool(get_value(propagation_parameters, "use_distance_mu", default=False))
    mu_def = float(get_value(propagation_parameters, "mu_def", default=1.0))
    mu_fix = float(get_value(propagation_parameters, "mu_fix", default=1.0))
    if use_distance_mu or (abs(mu_def - mu_fix) > 1e-12):
        node_weights = _compute_node_stiffness_profile(mesh, propagation_parameters)
        deformation = _solve_weighted_extension(
            mesh,
            boundary_displacement,
            propagation_parameters,
            node_weights=node_weights,
        )
        if deformation is None:
            return None
        return deformation

    nodes = mesh.node
    cells = mesh.cell
    if nodes is None or cells is None:
        return None

    boundary_nodes = get_boundary_nodes(mesh, propagation_parameters)
    if boundary_nodes is None or boundary_nodes.size == 0:
        return None

    fixed_nodes = get_fixed_nodes(propagation_parameters)
    dim = nodes.shape[1]

    scalar_space = LagrangeFESpace(mesh, p=1)
    tensor_space = TensorFunctionSpace(scalar_space, (dim, -1))
    total_dofs = tensor_space.number_of_global_dofs()
    node_count = nodes.shape[0]
    if node_count * dim != total_dofs:
        return None

    constrained_nodes = bm.unique(bm.concatenate((boundary_nodes, fixed_nodes)))
    boundary_values = bm.zeros((node_count, dim), dtype=float)
    if isinstance(boundary_displacement, Mapping):
        if boundary_displacement:
            keys = bm.asarray(list(boundary_displacement.keys()), dtype=int)
            values = bm.asarray([_as_coordinate_vector(value, dim) for value in boundary_displacement.values()], dtype=float)
            valid = (keys >= 0) & (keys < node_count)
            boundary_values[keys[valid]] = values[valid]
    else:
        array = bm.asarray(boundary_displacement, dtype=float)
        if array.ndim == 2 and array.shape[0] == node_count and array.shape[1] >= dim:
            boundary_values[:] = array[:, :dim]
        else:
            boundary_values[constrained_nodes] = _as_coordinate_vector(array, dim)
    boundary_values[fixed_nodes] = 0.0

    if tensor_space.dof_priority:
        displacement = bm.zeros((dim, node_count), dtype=float)
        displacement[:, constrained_nodes] = boundary_values[constrained_nodes].T
        boundary_mask = bm.zeros((dim, node_count), dtype=bool)
        boundary_mask[:, constrained_nodes] = True
    else:
        displacement = bm.zeros((node_count, dim), dtype=float)
        displacement[constrained_nodes] = boundary_values[constrained_nodes]
        boundary_mask = bm.zeros((node_count, dim), dtype=bool)
        boundary_mask[constrained_nodes] = True
    displacement = displacement.reshape(-1)
    boundary_mask = boundary_mask.reshape(-1)

    lame_lambda = float(get_value(propagation_parameters, "lambda_lame", default=1.0))
    shear_modulus = float(
        get_value(
            propagation_parameters,
            "mu_def",
            "shear_modulus",
            "mu",
            default=1.0,
        )
    )
    hypo = "plane_strain" if dim == 2 else "3D"
    q = get_value(propagation_parameters, "q", default=None)

    material = LinearElasticMaterial(
        name="mesh_deformation",
        lame_lambda=lame_lambda,
        shear_modulus=shear_modulus,
        hypo=hypo,
    )
    bform = BilinearForm(tensor_space)
    bform.add_integrator(LinearElasticityIntegrator(material, q=q))
    matrix = bform.assembly()
    rhs = bm.zeros(total_dofs, dtype=float)
    bc = DirichletBC(tensor_space, gd=displacement, threshold=boundary_mask)
    matrix, rhs = bc.apply(matrix, rhs)

    solver_name = get_value(propagation_parameters, "linear_solver", default="scipy")
    solution = fealpy_spsolve(matrix, rhs, solver=solver_name)
    solution = bm.asarray(solution, dtype=float).reshape(-1)

    nodal_displacement = bm.zeros_like(nodes, dtype=float)
    if tensor_space.dof_priority:
        nodal_displacement = solution.reshape(dim, node_count).T
    else:
        nodal_displacement = solution.reshape(node_count, dim)

    return bm.asarray(nodal_displacement, dtype=float)

def _apply_deformation_handler(
    current_mesh: Any,
    deformation: Any,
    propagation_parameters: Any,
) -> Any | None:
    """调用 cashocs 风格的变形 handler。"""
    handler = get_value(propagation_parameters, "deformation_handler")
    if handler is None:
        return None

    move_mesh = get_value(handler, "move_mesh")
    if not callable(move_mesh):
        return None

    if hasattr(handler, "mesh"):
        handler.mesh = current_mesh

    validated_a_priori = bool(get_value(propagation_parameters, "validated_a_priori", default=False))
    test_for_intersections = bool(
        get_value(propagation_parameters, "test_for_intersections", default=True)
    )
    moved = move_mesh(
        deformation,
        validated_a_priori=validated_a_priori,
        test_for_intersections=test_for_intersections,
    )
    if moved is False:
        return None
    if moved is True or moved is None:
        return current_mesh
    return moved

def propagate_mesh(
    current_mesh: Any,
    boundary_displacement: Any,
    geometry_contract: Any,
    propagation_parameters: Any,
    cache: Any = None,
) -> TrialState:
    """把边界位移传播到内部网格。"""
    reference_mesh = deepcopy(current_mesh)
    trial_mesh = _copy_mesh(current_mesh)
    extension_path: str | None = None
    used_deformation_extension = False
    deformation_extension_result: Any = None
    deformation = _solve_linear_elasticity_extension(trial_mesh, boundary_displacement, propagation_parameters)
    if deformation is not None:
        deformation_limits = _check_deformation_limits(trial_mesh, deformation, propagation_parameters)
        if not deformation_limits["accepted"]:
            return TrialState(
                trial_mesh=reference_mesh,
                trial_design_boundary=boundary_displacement,
                trial_boundary_displacement=boundary_displacement,
                trial_cache=dict(cache) if isinstance(cache, Mapping) else cache,
                trial_objective=None,
                quality_info={
                    "status": "rejected",
                    "accepted": False,
                    "reason": "deformation_limits",
                    **deformation_limits,
                },
                extension_path="linear_elasticity",
                used_deformation_extension=True,
                restart_optimization=False,
                deformation_extension_result=deformation,
            )
        handler_mesh = _apply_deformation_handler(trial_mesh, deformation, propagation_parameters)
        if handler_mesh is not None:
            trial_mesh = handler_mesh
            extension_path = "deformation_handler"
            used_deformation_extension = True
            deformation_extension_result = handler_mesh
        else:
            nodes = trial_mesh.node
            if nodes is not None:
                trial_mesh = _set_mesh_nodes(trial_mesh, nodes + deformation)
            extension_path = "linear_elasticity"
            used_deformation_extension = True
            deformation_extension_result = trial_mesh
    else:
        extension = get_value(propagation_parameters, "deformation_extension", "deformation_handler")
        extended_mesh = extension(trial_mesh, boundary_displacement, geometry_contract, propagation_parameters, cache=cache) if callable(extension) else None
        if extended_mesh is None:
            extended_mesh = _update_boundary_coordinates(trial_mesh, boundary_displacement, geometry_contract)
            extension_path = "boundary_coordinates"
            used_deformation_extension = False
        else:
            extension_path = "deformation_extension"
            used_deformation_extension = True
        trial_mesh = extended_mesh
        deformation_extension_result = extended_mesh

    quality_info = check_mesh_quality(trial_mesh, geometry_contract, propagation_parameters)
    if isinstance(quality_info, Mapping) and not quality_info.get("accepted", True):
        trial_mesh = reference_mesh

    trial_cache = dict(cache) if isinstance(cache, Mapping) else cache
    return TrialState(
        trial_mesh=trial_mesh,
        trial_design_boundary=boundary_displacement,
        trial_boundary_displacement=boundary_displacement,
        trial_cache=trial_cache,
        trial_objective=None,
        quality_info=quality_info,
        extension_path=extension_path,
        used_deformation_extension=used_deformation_extension,
        restart_optimization=False,
        deformation_extension_result=deformation_extension_result,
    )

def check_mesh_quality(
    mesh: Any,
    geometry_contract: Any = None,
    propagation_parameters: Any = None,
) -> Any:
    """检查网格质量。"""
    quality_measure, quality_type, quality_quantile, tol_lower, tol_upper = _mesh_quality_parameters(
        propagation_parameters
    )
    test_for_intersections = bool(get_value(propagation_parameters, "test_for_intersections", default=True))
    remesh_iter = int(get_value(propagation_parameters, "remesh_iter", default=0))

    def _quality_record(quality_value: Any, has_negative_cells: bool) -> dict[str, Any]:
        quality_value = float(quality_value)
        state = _classify_mesh_quality(
            quality_value,
            tol_lower,
            tol_upper,
            has_negative_cells=has_negative_cells,
            test_for_intersections=test_for_intersections,
        )
        needs_remesh = bool(remesh_iter) and state["status"] == "marginal"
        return {
            "status": state["status"],
            "quality_state": state["quality_state"],
            "accepted": state["accepted"],
            "needs_remesh": needs_remesh,
            "reason": state["reason"],
            "quality": quality_value,
            "quality_measure": quality_measure,
            "quality_type": quality_type,
            "quality_quantile": quality_quantile,
            "min_cell_measure": quality_value,
            "max_cell_measure": quality_value,
            "mean_cell_measure": quality_value,
            "has_negative_cells": has_negative_cells,
            "test_for_intersections": test_for_intersections,
            "tol_lower": tol_lower,
            "tol_upper": tol_upper,
        }

    cell_qualities = _compute_cell_quality_values(mesh, quality_measure)
    quality = compute_mesh_quality(
        mesh,
        quality_type=quality_type,
        quality_measure=quality_measure,
        quantile=quality_quantile,
    )
    nodes = mesh.node
    cells = mesh.cell
    if cell_qualities is not None:
        has_negative_cells = False
        signed_area = None
        if nodes is not None and cells is not None and cells.shape[1] == 3 and nodes.shape[1] >= 2:
            triangles = bm.asarray(cells, dtype=int)
            coords = bm.asarray(nodes, dtype=float)
            v0 = coords[triangles[:, 0]]
            v1 = coords[triangles[:, 1]]
            v2 = coords[triangles[:, 2]]
            signed_area = 0.5 * ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - 
                                 (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
            has_negative_cells = bool(bm.any(signed_area <= 0.0))
        result = _quality_record(quality, has_negative_cells)
        result["cell_qualities"] = bm.asarray(cell_qualities, dtype=float)
        result["min_cell_measure"] = float(bm.min(cell_qualities)) if cell_qualities.size > 0 else 0.0
        result["max_cell_measure"] = float(bm.max(cell_qualities)) if cell_qualities.size > 0 else 0.0
        result["mean_cell_measure"] = float(bm.mean(cell_qualities)) if cell_qualities.size > 0 else 0.0
        if signed_area is not None:
            result["signed_cell_measure"] = bm.asarray(signed_area, dtype=float)
        return result
    if isinstance(mesh, Mapping) and "quality" in mesh:
        value = mesh["quality"]
        if isinstance(value, Mapping):
            return value
        quality_value = float(value)
        if propagation_parameters is None:
            return quality_value
        return _quality_record(quality_value, False)
    if hasattr(mesh, "quality"):
        quality = getattr(mesh, "quality")
        if callable(quality):
            quality = quality()
        if isinstance(quality, Mapping):
            return quality
        if propagation_parameters is None:
            return quality
        return _quality_record(quality, False)
    return {"status": "unknown", "accepted": True}

def remesh_mesh(
    current_mesh: Any,
    geometry_contract: Any,
    propagation_parameters: Any,
    cache: Any = None,
) -> RemeshResult:
    """构造重网格审计结果。"""
    remesh_handler = get_value(propagation_parameters, "remesh_handler", "remesh")
    if not callable(remesh_handler):
        return RemeshResult(remeshed_state=_copy_mesh(current_mesh), remeshed=False, requires_remesh=False, restart_optimization=False, quality_info={"status": "skipped"}, reason="no_remesh_handler")

    remesh_output = remesh_handler(current_mesh, geometry_contract, propagation_parameters, cache=cache)
    if hasattr(remesh_output, "remeshed_state"):
        return remesh_output
    if remesh_output is False:
        return RemeshResult(remeshed_state=current_mesh, remeshed=False, requires_remesh=True, restart_optimization=False, quality_info={"status": "rejected", "reason": "remesh_handler_returned_false"}, remesh_output=remesh_output, reason="handler_rejected")
    return RemeshResult(remeshed_state=current_mesh if remesh_output is True or remesh_output is None else remesh_output, remeshed=True, requires_remesh=False, restart_optimization=True, quality_info={"status": "ok"}, handler_name=_handler_name(remesh_handler), remesh_output=remesh_output, reason="handler_succeeded")


build_trial_mesh = propagate_mesh
