"""几何梯度与更新骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm, BoundaryFaceSourceIntegrator, DirichletBC, LinearElasticityIntegrator, LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material import LinearElasticMaterial
from fealpy.solver import spsolve
from .geometry_regularization import _polygon_vertex_normals,_polygon_area_and_centroid

from .benchmark_common import get_value

@dataclass(slots=True)
class GeometryGradientResult:
    """几何梯度结果。"""

    raw_gradient: Any
    normal_gradient: Any
    node_gradient: Any
    descent_direction: Any
    propagated_gradient: Any
    shape_derivative: Any = None
    boundary_normal_component: Any = None
    riesz_projection: Any = None


def _mesh_dimension(mesh: Any, geometry_contract: Any = None) -> int | None:
    """提取空间维度。"""
    value = get_value(geometry_contract, "spatial_dim", default=None)
    if value is not None:
        return int(value)
    nodes = mesh.node
    if nodes is not None and nodes.ndim >= 2:
        return int(nodes.shape[1])
    return None


def _cache_propagation_parameters(cache: Any, options: Any = None) -> Mapping[str, Any]:
    """提取传播参数。"""
    for source in (cache, options):
        value = get_value(source, "propagation_parameters", default=None)
        if isinstance(value, Mapping):
            return value
    return {}


def _boundary_node_ids(cache: Any, geometry_contract: Any, options: Any = None) -> Any:
    """提取设计边界节点。"""
    for source in (_cache_propagation_parameters(cache, options), cache, options):
        value = get_value(source, "design_boundary_nodes", "boundary_nodes", "shape_bdry_def", default=None)
        if value is not None:
            return bm.asarray(value, dtype=int)
    value = get_value(geometry_contract, "design_boundary_ids", default=())
    return bm.asarray(value, dtype=int)


def _fixed_node_ids(cache: Any, geometry_contract: Any, options: Any = None) -> Any:
    """提取固定边界节点。"""
    for source in (_cache_propagation_parameters(cache, options), cache, options):
        value = get_value(source, "fixed_nodes", "shape_bdry_fix", default=None)
        if value is not None:
            return bm.asarray(value, dtype=int)
    value = get_value(geometry_contract, "fixed_boundary_ids", default=())
    return bm.asarray(value, dtype=int)


def _coerce_vector(value: Any, dim: int, normal: Any = None) -> Any:
    """把标量或序列转成定长向量。"""
    if value is None:
        return bm.zeros(dim, dtype=float)
    if isinstance(value, Mapping):
        arr = bm.asarray([value[key] for key in sorted(value)], dtype=float).reshape(-1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        arr = bm.asarray(list(value), dtype=float).reshape(-1)
    else:
        arr = bm.asarray(value, dtype=float).reshape(-1)
    if arr.size == dim:
        return arr.astype(float, copy=False)
    if arr.size == 1:
        scalar = float(arr.reshape(()))
        if normal is not None:
            normal_vec = bm.asarray(normal, dtype=float).reshape(-1)
            if normal_vec.size == dim:
                return scalar * normal_vec
        return bm.full(dim, scalar, dtype=float)
    if normal is not None:
        normal_vec = bm.asarray(normal, dtype=float).reshape(-1)
        if normal_vec.size == dim:
            return float(bm.dot(arr[:dim], normal_vec)) * normal_vec
    raise ValueError("gradient vector size mismatch")


def get_normal_info(cache: Any, node: int, default: Any = None) -> Any:
    """查找节点法向信息。"""
    normal_info = get_value(cache, "normal_info", "boundary_normals", default=None)
    if normal_info is None:
        return default
    if isinstance(normal_info, Mapping):
        return normal_info.get(int(node), default)
    node_index = int(node)
    if 0 <= node_index < len(normal_info):
        return normal_info[node_index]
    return default


def _boundary_normal_unit_vectors(
    node_ids: Any,
    cache: Any,
    geometry_contract: Any,
    *,
    mesh: Any = None,
) -> Any | None:
    """提取边界节点的单位法向。"""
    node_ids = bm.asarray(node_ids, dtype=int).reshape(-1)
    if node_ids.size == 0:
        return None

    dim = _mesh_dimension(mesh, geometry_contract)
    reference_mesh = get_value(cache, "reference_mesh", default=None)
    if dim is None:
        dim = _mesh_dimension(reference_mesh, geometry_contract)
    if dim is None:
        return None

    normal_info = get_value(cache, "normal_info", "boundary_normals", default=None)
    if isinstance(normal_info, Mapping):
        normal_keys = bm.asarray(list(normal_info.keys()), dtype=int)
        normal_values = bm.asarray([bm.asarray(value, dtype=float).reshape(-1) for value in normal_info.values()], dtype=float)
        order = bm.argsort(normal_keys)
        normal_keys = normal_keys[order]
        normal_values = normal_values[order]
        positions = bm.searchsorted(normal_keys, node_ids)
        valid = positions < normal_keys.size
        if bm.any(valid):
            matched = bm.zeros_like(valid)
            valid_positions = positions[valid]
            matched[valid] = normal_keys[valid_positions] == node_ids[valid]
            valid &= matched
        if bm.all(valid):
            norms = bm.linalg.norm(normal_values[positions], axis=1)
            if bm.all(norms > 0.0):
                return normal_values[positions] / norms[:, None]

    if isinstance(normal_info, Sequence) and not isinstance(normal_info, (str, bytes, bytearray)):
        normal_values = bm.asarray(normal_info, dtype=float)
        if normal_values.ndim >= 2 and normal_values.shape[0] > int(bm.max(node_ids)):
            normals = bm.asarray(normal_values[node_ids], dtype=float)
            norms = bm.linalg.norm(normals, axis=1)
            if bm.all(norms > 0.0):
                return normals / norms[:, None]

    if dim != 2:
        return None

    nodes = mesh.node
    if nodes is None:
        nodes = reference_mesh.node
    if nodes is None:
        return None
    if bm.any(node_ids < 0) or bm.any(node_ids >= nodes.shape[0]):
        return None

    coords = bm.asarray(nodes[node_ids], dtype=float)
    if coords.shape[0] == 0:
        return None
    return _polygon_vertex_normals(coords)


def _project_value_to_normal(value: Any, normal: Any, dim: int) -> tuple[float, ...]:
    """投影到法向。"""
    vec = _coerce_vector(value, dim, normal=normal)
    normal_vec = bm.asarray(normal, dtype=float).reshape(-1)
    if normal_vec.size != dim:
        return tuple(float(component) for component in bm.asarray(vec, dtype=float).reshape(-1))
    norm = float(bm.linalg.norm(normal_vec))
    if norm <= 0.0:
        return tuple(float(component) for component in bm.asarray(vec, dtype=float).reshape(-1))
    unit_normal = normal_vec / norm
    projected = float(bm.dot(vec, unit_normal)) * unit_normal
    return tuple(float(component) for component in bm.asarray(projected, dtype=float).reshape(-1))


def _normalize_boundary_vector_map(
    source: Any,
    node_ids: Any,
    dim: int,
    cache: Any = None,
    project_normals: bool = False,
) -> dict[int, Any]:
    """把边界导数整理成节点向量映射。"""
    if source is None:
        return {int(node): bm.zeros(dim, dtype=float) for node in node_ids.tolist()}
    if isinstance(source, Mapping):
        return {
            int(node): _project_value_to_normal(source[int(node)], normal, dim)
            if project_normals and (normal := get_normal_info(cache, int(node), default=None)) is not None
            else _coerce_vector(source[int(node)], dim, normal=normal)
            for node in node_ids.tolist()
            if int(node) in source
        }
    arr = bm.asarray(source, dtype=float)
    if arr.ndim == 1 and arr.size == dim:
        return {int(node): arr.astype(float, copy=True) for node in node_ids.tolist()}
    if arr.ndim >= 2 and arr.shape[-1] == dim:
        flat = arr.reshape(-1, dim)
        return {
            int(node): flat[idx].astype(float, copy=True)
            for idx, node in enumerate(node_ids.tolist())
            if idx < flat.shape[0]
        }
    return {int(node): bm.full(dim, float(arr.reshape(())), dtype=float) for node in node_ids.tolist()}


def _build_nodal_vector_array(
    mesh: Any,
    nodal_map: Mapping[int, Any],
    dim: int,
) -> Any | None:
    """把节点映射转成完整向量场数组。"""
    nodes = mesh.node
    if nodes is None:
        return None
    nodal_values = bm.zeros((nodes.shape[0], dim), dtype=float)
    if nodal_map:
        node_ids = bm.asarray(list(nodal_map.keys()), dtype=int)
        values = bm.asarray([_coerce_vector(value, dim) for value in nodal_map.values()], dtype=float)
        valid = (node_ids >= 0) & (node_ids < nodal_values.shape[0])
        nodal_values[node_ids[valid]] = values[valid]
    return nodal_values


def _vector_array_to_nodal_map(array: Any) -> dict[int, tuple[float, ...]]:
    """把完整向量场数组转成节点映射。"""
    values = bm.asarray(array, dtype=float)
    if values.ndim == 1:
        return {int(i): (float(value),) for i, value in enumerate(values)}
    return {
        int(i): tuple(float(component) for component in row)
        for i, row in enumerate(values)
    }
    
def _add_gradient_maps(left: Any, right: Any) -> Any:
    """Add two gradient-like objects while preserving mapping structure."""
    if left is None:
        return right
    if right is None:
        return left

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        result = {}
        for key in keys:
            lv = bm.asarray(left.get(key, 0.0), dtype=float).reshape(-1)
            rv = bm.asarray(right.get(key, 0.0), dtype=float).reshape(-1)

            if lv.size == 1 and rv.size > 1:
                lv = bm.full(rv.shape, float(lv[0]), dtype=float)
            if rv.size == 1 and lv.size > 1:
                rv = bm.full(lv.shape, float(rv[0]), dtype=float)

            if lv.shape != rv.shape:
                size = max(lv.size, rv.size)
                if lv.size < size:
                    lv = bm.pad(lv, (0, size - lv.size))
                if rv.size < size:
                    rv = bm.pad(rv, (0, size - rv.size))

            value = lv + rv
            result[int(key)] = tuple(float(v) for v in value)
        return result

    if isinstance(left, Mapping):
        result = dict(left)
        right_value = bm.asarray(right, dtype=float).reshape(-1)
        for key, value in result.items():
            lv = bm.asarray(value, dtype=float).reshape(-1)
            rv = right_value
            if rv.size == 1 and lv.size > 1:
                rv = bm.full(lv.shape, float(rv[0]), dtype=float)
            result[int(key)] = tuple(float(v) for v in lv + rv)
        return result

    if isinstance(right, Mapping):
        return _add_gradient_maps(right, left)

    try:
        return bm.asarray(left, dtype=float) + bm.asarray(right, dtype=float)
    except Exception:
        return right

def _polygon_area_gradient(points: Any) -> Any:
    """Compute the discrete area gradient for an ordered 2D polygon."""
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return bm.zeros_like(coords, dtype=float)
    if coords.shape[0] == 1:
        return bm.zeros_like(coords, dtype=float)
    prev_coords = bm.roll(coords, 1, axis=0)
    next_coords = bm.roll(coords, -1, axis=0)
    return 0.5 * bm.stack(
        [
            next_coords[:, 1] - prev_coords[:, 1],
            prev_coords[:, 0] - next_coords[:, 0],
        ],
        axis=1,
    )

def _design_boundary_face_ids(mesh: Any, design_node_ids: Any) -> Any:
    """根据设计边界节点，推断设计边界面编号。"""
    nodes = mesh.node
    if nodes is None or design_node_ids.size == 0:
        return bm.zeros(0, dtype=int)
    faces = mesh.entity("face") if hasattr(mesh, "entity") else getattr(mesh, "face", None)
    if faces is None:
        return bm.zeros(0, dtype=int)
    faces = bm.asarray(faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] < 2:
        return bm.zeros(0, dtype=int)
    design_set = bm.asarray(design_node_ids, dtype=int)
    mask = bm.all(bm.isin(faces[:, :2], design_set), axis=1)
    return bm.nonzero(mask)[0].astype(int, copy=False)


class _BoundaryShapeDerivativeSource:
    """把节点形状导数离散成边界面上的源项。"""

    coordtype = "cartesian"

    def __init__(self, mesh: Any, nodal_map: Mapping[int, Any], design_node_ids: Any, dim: int) -> None:
        self.mesh = mesh
        self.dim = dim
        self.node_coords = mesh.node
        self.design_node_ids = bm.asarray(design_node_ids, dtype=int)
        self.values = {
            int(node): _coerce_vector(value, dim)
            for node, value in nodal_map.items()
        }

    def __call__(self, points: Any, normal: Any = None) -> Any:
        coords = bm.asarray(points, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if self.node_coords is None or self.design_node_ids.size == 0:
            return bm.zeros(coords.shape[:-1] + (self.dim,), dtype=float)
        design_coords = self.node_coords[self.design_node_ids]
        design_values = bm.asarray([self.values.get(int(node), bm.zeros(self.dim, dtype=float)) for node in self.design_node_ids.tolist()], dtype=float)
        flat_coords = coords.reshape(-1, coords.shape[-1])
        distances = bm.linalg.norm(design_coords[None, :, :] - flat_coords[:, None, :], axis=2)
        nearest = bm.argmin(distances, axis=1)
        result = design_values[nearest]
        return result.reshape(coords.shape[:-1] + (self.dim,))


def _vector_space_total_dofs(vector_space: Any, dim: int, node_count: int) -> int | None:
    """提取向量空间总自由度数。"""
    total = int(vector_space.number_of_global_dofs())
    if total != dim * node_count:
        return None
    return total


def _vector_dof_mask(vector_space: Any, node_ids: Any, dim: int, node_count: int) -> Any:
    """为给定节点构造自由度掩码。"""
    total = int(vector_space.number_of_global_dofs())
    mask = bm.zeros(total, dtype=bool)
    node_ids = bm.asarray(node_ids, dtype=int).reshape(-1)
    if node_ids.size == 0:
        return mask
    if getattr(vector_space, "dof_priority", False):
        indices = (bm.arange(dim, dtype=int)[:, None] * node_count) + node_ids[None, :]
    else:
        indices = (node_ids[:, None] * dim) + bm.arange(dim, dtype=int)[None, :]
    mask[indices.reshape(-1)] = True
    return mask


def _vector_dof_values(vector_space: Any, nodal_map: Mapping[int, Any], dim: int, node_count: int) -> Any:
    """把节点映射转成向量空间自由度值。"""
    total = int(vector_space.number_of_global_dofs())
    values = bm.zeros(total, dtype=float)
    if not nodal_map:
        return values
    node_ids = bm.asarray(list(nodal_map.keys()), dtype=int).reshape(-1)
    vecs = bm.asarray([_coerce_vector(item, dim) for item in nodal_map.values()], dtype=float)
    valid = (node_ids >= 0) & (node_ids < node_count)
    node_ids = node_ids[valid]
    vecs = vecs[valid]
    if node_ids.size == 0:
        return values
    if getattr(vector_space, "dof_priority", False):
        arrays = bm.zeros((dim, node_count), dtype=float)
        arrays[:, node_ids] = vecs.T
        return arrays.reshape(-1)
    arrays = bm.zeros((node_count, dim), dtype=float)
    arrays[node_ids] = vecs
    return arrays.reshape(-1)


def _elasticity_parameters(options: Any, cache: Any, geometry_contract: Any) -> tuple[float, float, str, Any]:
    """提取线弹性参数。"""
    propagation_parameters = _cache_propagation_parameters(cache, options)
    lame_lambda = float(get_value(propagation_parameters, "lambda_lame", "lame_lambda", default=get_value(options, "lambda_lame", default=1.0)))
    mu_value = float(get_value(propagation_parameters, "mu_def", "shear_modulus", "mu", default=get_value(options, "mu_def", default=1.0)))
    hypo = get_value(propagation_parameters, "hypo", default=get_value(options, "hypo", default="plane_strain"))
    q = get_value(options, "q", default=get_value(propagation_parameters, "q", default=None))
    return lame_lambda, mu_value, hypo, q


def _solve_linear_elasticity_projection(
    mesh: Any,
    nodal_source: Mapping[int, Any],
    geometry_contract: Any,
    cache: Any,
    options: Any = None,
    *,
    boundary_values: Mapping[int, Any] | None = None,
) -> Any | None:
    """用线弹性方程把节点源项投影为平滑向量场。"""
    nodes = mesh.node
    dim = _mesh_dimension(mesh, geometry_contract)
    if nodes is None or dim is None:
        return None

    scalar_space = LagrangeFESpace(mesh, p=1)
    vector_space = TensorFunctionSpace(scalar_space, (dim, -1))
    total_dofs = _vector_space_total_dofs(vector_space, dim, nodes.shape[0])
    if total_dofs is None:
        return None

    lame_lambda, mu_value, hypo, q = _elasticity_parameters(options, cache, geometry_contract)
    material = LinearElasticMaterial(
        name="shape_gradient_projection",
        lame_lambda=lame_lambda,
        shear_modulus=mu_value,
        hypo=hypo,
        device=bm.get_device(nodes),
    )

    bform = BilinearForm(vector_space)
    bform.add_integrator(LinearElasticityIntegrator(material, q=q))
    matrix = bform.assembly()
    rhs = None
    use_boundary_face_source = bool(get_value(options, "use_boundary_face_source", default=False))
    design_face_ids = bm.zeros(0, dtype=int)
    if use_boundary_face_source and isinstance(nodal_source, Mapping):
        design_nodes = bm.asarray(sorted(int(node) for node in nodal_source.keys()), dtype=int)
        design_face_ids = _design_boundary_face_ids(mesh, design_nodes)
        if design_face_ids.size > 0:
            source = _BoundaryShapeDerivativeSource(mesh, nodal_source, design_nodes, dim)
            rhs_form = LinearForm(vector_space)
            rhs_form.add_integrator(
                BoundaryFaceSourceIntegrator(
                    source=source,
                    q=q,
                    threshold=design_face_ids,
                )
            )
            rhs = rhs_form.assembly()

    if rhs is None:
        rhs = bm.zeros(total_dofs, dtype=float)
        rhs[:] = _vector_dof_values(vector_space, nodal_source, dim, nodes.shape[0])

    fixed_nodes = _fixed_node_ids(cache, geometry_contract, options)
    if fixed_nodes.size > 0:
        boundary_mask = _vector_dof_mask(vector_space, fixed_nodes, dim, nodes.shape[0])
        bc = DirichletBC(
            vector_space,
            gd=0.0,
            threshold=boundary_mask,
        )
        matrix, rhs = bc.apply(matrix, rhs)

    solution = spsolve(matrix, rhs, solver=get_value(options, "linear_solver", default="scipy"))
    solution = bm.asarray(solution, dtype=float).reshape(-1)
    return solution.reshape(nodes.shape[0], dim) if getattr(vector_space, "dof_priority", False) is False else _solution_to_priority_array(solution, dim, nodes.shape[0])


def _solution_to_priority_array(solution: Any, dim: int, node_count: int) -> Any:
    """把优先级自由度解转成节点-分量数组。"""
    return bm.asarray(solution, dtype=float).reshape(dim, node_count).T


def _zero_fixed_nodes_in_mapping(
    values: Mapping[int, Any],
    fixed_node_ids: Any,
) -> dict[int, Any]:
    """将固定节点的梯度分量清零。"""
    fixed = {int(node) for node in fixed_node_ids.tolist()}

    def _zero_like(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: 0.0 for key in value}
        array = bm.asarray(value, dtype=float)
        if array.ndim == 0:
            return 0.0
        zeroed = bm.zeros_like(array, dtype=float)
        if isinstance(value, tuple):
            return tuple(float(item) for item in zeroed.reshape(-1))
        if isinstance(value, list):
            return [float(item) for item in zeroed.reshape(-1)]
        return zeroed

    return {
        int(node): (_zero_like(value) if int(node) in fixed else value)
        for node, value in values.items()
    }


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    """从映射或对象属性里取第一个可用字段。"""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return next((source[name] for name in names if name in source), default)
    return next((getattr(source, name) for name in names if hasattr(source, name)), default)


def _sum_scalar_value(value: Any) -> float:
    """递归求和一个结构中的标量分量。"""
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, Mapping):
        return float(sum(bm.asarray([_sum_scalar_value(item) for item in value.values()], dtype=float)))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(sum(bm.asarray([_sum_scalar_value(item) for item in value], dtype=float)))
    return 0.0


def _negate_value(value: Any) -> Any:
    """取数值或映射的相反数。"""
    if isinstance(value, Number):
        return -value
    if isinstance(value, Mapping):
        return {key: _negate_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, tuple):
            return tuple(_negate_value(item) for item in value)
        return type(value)(_negate_value(item) for item in value)
    return value


def _scale_value(value: Any, factor: float) -> Any:
    """缩放数值或映射。"""
    if isinstance(value, Number):
        return factor * value
    if isinstance(value, Mapping):
        return {key: _scale_value(item, factor) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, tuple):
            return tuple(_scale_value(item, factor) for item in value)
        return type(value)(_scale_value(item, factor) for item in value)
    return value


def _dot_value(left: Any, right: Any) -> float:
    """计算两个梯度表示的方向导数。"""
    if isinstance(left, Number) and isinstance(right, Number):
        return float(left) * float(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return float(sum(bm.asarray([_dot_value(value, right.get(key, 0.0)) for key, value in left.items()], dtype=float)))
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes, bytearray)):
        return float(sum(bm.asarray([_dot_value(left_item, right_item) for left_item, right_item in zip(left, right)], dtype=float)))
    if isinstance(left, Mapping) and isinstance(right, Number):
        return float(right) * float(sum(bm.asarray([_sum_scalar_value(value) for value in left.values()], dtype=float)))
    if isinstance(left, Number) and isinstance(right, Mapping):
        return float(left) * float(sum(bm.asarray([_sum_scalar_value(value) for value in right.values()], dtype=float)))
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)) and isinstance(right, Number):
        return float(right) * float(sum(bm.asarray([_sum_scalar_value(value) for value in left], dtype=float)))
    if isinstance(left, Number) and isinstance(right, Sequence) and not isinstance(right, (str, bytes, bytearray)):
        return float(left) * float(sum(bm.asarray([_sum_scalar_value(value) for value in right], dtype=float)))
    return 0.0


def assemble_geometry_gradient(
    mesh: Any,
    state_result: Any,
    adjoint_result: Any,
    geometry_contract: Any,
    cache: Any,
    objective_result: Any,
    options: Any = None,
) -> GeometryGradientResult:
    """组装几何梯度。"""
    propagation_parameters = _cache_propagation_parameters(cache, options)
    raw_gradient = get_value(adjoint_result, "gradient", "raw_gradient")
    if raw_gradient is None:
        raw_gradient = get_value(objective_result, "gradient", "raw_gradient")

    shape_derivative = get_value(
        objective_result,
        "shape_derivative",
        "raw_shape_derivative",
        default=None,
    )
    if shape_derivative is None:
        shape_derivative = get_value(
            adjoint_result,
            "shape_derivative",
            "raw_shape_derivative",
            default=None,
        )
    
    objective_shape_derivative = get_value(
    objective_result,
    "shape_derivative",
    "raw_shape_derivative",
    default=None,
    )
    adjoint_shape_derivative = get_value(
        adjoint_result,
        "shape_derivative",
        "raw_shape_derivative",
        default=None,
    )
    shape_derivative = _add_gradient_maps(
        adjoint_shape_derivative,
        objective_shape_derivative,
    )
    
    if shape_derivative is None and raw_gradient is not None:
        shape_derivative = raw_gradient
    if shape_derivative is None:
        shape_derivative = get_value(objective_result, "total_objective")
    if raw_gradient is None:
        raw_gradient = shape_derivative

    fixed_node_ids = _fixed_node_ids(cache, geometry_contract, options)
    dim = _mesh_dimension(mesh, geometry_contract)
    use_real_projection = mesh is not None and mesh.node is not None and dim is not None

    boundary_normal_component = project_to_boundary_normals(
        shape_derivative,
        geometry_contract,
        cache,
        options,
    )
    if isinstance(boundary_normal_component, Mapping) and fixed_node_ids.size > 0:
        boundary_normal_component = _zero_fixed_nodes_in_mapping(boundary_normal_component, fixed_node_ids)

    # cashocs 的主路径是：shape derivative -> Riesz 投影 ->（可选）边界重扩展。
    # 因此这里的线弹性投影必须直接吃 shape derivative，而不是先把源项压成法向分量。
    riesz_projection = filter_fixed_boundaries(
        shape_derivative,
        geometry_contract,
        options,
    )
    propagated_gradient = riesz_projection
    if use_real_projection:
        design_nodes = _boundary_node_ids(cache, geometry_contract, options)
        project_source = shape_derivative
        if not isinstance(project_source, Mapping):
            project_source = _normalize_boundary_vector_map(
                project_source,
                design_nodes,
                dim,
                cache=cache,
                project_normals=False,
            )
        elif not project_source:
            project_source = _normalize_boundary_vector_map(
                shape_derivative,
                design_nodes,
                dim,
                cache=cache,
                project_normals=False,
            )

        projected = _solve_linear_elasticity_projection(
            mesh,
            project_source,
            geometry_contract,
            cache,
            options,
        )
        if projected is not None:
            riesz_projection = _vector_array_to_nodal_map(projected)
            if fixed_node_ids.size > 0:
                riesz_projection = _zero_fixed_nodes_in_mapping(riesz_projection, fixed_node_ids)
            propagated_gradient = riesz_projection

    normal_gradient = boundary_normal_component
    node_gradient = propagated_gradient
    descent_source = node_gradient
    if descent_source is None:
        descent_source = normal_gradient
    if descent_source is None:
        descent_source = raw_gradient
    descent_direction = _scale_value(descent_source, -1.0) if descent_source is not None else None

    return GeometryGradientResult(
        raw_gradient=raw_gradient,
        normal_gradient=normal_gradient,
        node_gradient=node_gradient,
        descent_direction=descent_direction,
        propagated_gradient=propagated_gradient,
        shape_derivative=shape_derivative,
        boundary_normal_component=boundary_normal_component,
        riesz_projection=riesz_projection,
    )


def project_to_boundary_normals(
    raw_gradient: Any,
    geometry_contract: Any,
    cache: Any,
    options: Any = None,
) -> Any:
    """投影到边界法向。"""
    provider = get_value(options, "project_to_boundary_normals")
    if callable(provider):
        return provider(raw_gradient, geometry_contract, cache, options=options)
    normal_info = get_value(cache, "normal_info", "boundary_normals", default=None)
    if isinstance(raw_gradient, Mapping) and normal_info is not None:
        projected = {}
        dim = _mesh_dimension(None, geometry_contract)
        if isinstance(normal_info, Mapping):
            for key, value in raw_gradient.items():
                normal = normal_info.get(int(key))
                projected[key] = value if normal is None else _project_value_to_normal(value, normal, dim or len(bm.asarray(normal).reshape(-1)))
            return projected
        if isinstance(normal_info, Sequence) and not isinstance(normal_info, (str, bytes, bytearray)):
            normal_count = len(normal_info)
            for key, value in raw_gradient.items():
                node_index = int(key)
                if node_index < 0 or node_index >= normal_count:
                    projected[key] = value
                    continue
                normal = normal_info[node_index]
                projected[key] = _project_value_to_normal(value, normal, dim or len(bm.asarray(normal).reshape(-1)))
            return projected
    return raw_gradient


def filter_fixed_boundaries(
    node_gradient: Any,
    geometry_contract: Any,
    options: Any = None,
) -> Any:
    """过滤固定边界上的分量。"""
    fixed_boundary_ids = set(getattr(geometry_contract, "fixed_boundary_ids", ()) or ())
    if not fixed_boundary_ids or not isinstance(node_gradient, Mapping):
        return node_gradient
    return {
        key: (0.0 if key in fixed_boundary_ids else value)
        for key, value in node_gradient.items()
    }


def compute_descent_direction(
    geometry_gradient: GeometryGradientResult,
    geometry_contract: Any,
    options: Any = None,
) -> Any:
    """计算下降方向。"""
    descent_direction = getattr(geometry_gradient, "descent_direction", None)
    if descent_direction is not None:
        return descent_direction
    node_gradient = getattr(geometry_gradient, "node_gradient", None)
    if node_gradient is not None:
        return _scale_value(node_gradient, -1.0)
    normal_gradient = getattr(geometry_gradient, "normal_gradient", geometry_gradient)
    return _scale_value(normal_gradient, -1.0)


def build_boundary_displacement(
    descent_direction: Any,
    step_size: float,
    geometry_contract: Any,
    options: Any = None,
    mesh: Any = None,
    cache: Any = None,
    objective_parameters: Any = None,
) -> Any:
    """构造边界位移。"""
    projected_direction = project_to_area_preserving_tangent_space(
        descent_direction,
        geometry_contract,
        cache,
        options,
        mesh=mesh,
    )
    return _scale_value(projected_direction, step_size)


def build_trial_update(
    current_state: Any,
    cache: Any,
    descent_direction: Any,
    step_size: float,
    geometry_contract: Any,
    options: Any = None,
    mesh: Any = None,
) -> Any:
    """构造试探更新。"""
    boundary_displacement = build_boundary_displacement(
        descent_direction,
        step_size,
        geometry_contract,
        options,
        mesh=mesh,
        cache=cache,
    )
    return {
        "current_state": current_state,
        "cache": cache,
        "boundary_displacement": boundary_displacement,
        "step_size": step_size,
        "geometry_contract": geometry_contract,
        "options": options,
    }


def project_to_area_preserving_tangent_space(
    raw_gradient: Any,
    geometry_contract: Any,
    cache: Any,
    options: Any = None,
    *,
    mesh: Any = None,
) -> Any:
    """Project a 2D boundary displacement onto the area-preserving tangent space."""
    provider = get_value(options, "project_to_area_preserving_tangent_space")
    if callable(provider):
        return provider(raw_gradient, geometry_contract, cache, options=options, mesh=mesh)

    if not bool(get_value(options, "area_preserving_projection", default=False)):
        return raw_gradient

    dim = _mesh_dimension(mesh, geometry_contract)
    if dim != 2:
        return raw_gradient

    design_node_ids = _boundary_node_ids(cache, geometry_contract, options)
    if design_node_ids.size < 3:
        return raw_gradient

    nodes = mesh.node
    if nodes is None:
        reference_mesh = get_value(cache, "reference_mesh", default=None)
        if reference_mesh is not None:
            nodes = reference_mesh.node
    if nodes is None:
        return raw_gradient

    design_node_ids = bm.asarray(design_node_ids, dtype=int)
    if bm.any(design_node_ids < 0) or bm.any(design_node_ids >= nodes.shape[0]):
        return raw_gradient

    node_coords = bm.asarray(nodes[design_node_ids], dtype=float)
    if node_coords.shape[0] < 3:
        return raw_gradient

    preserve_normal_direction = bool(
        get_value(
            options,
            "area_preserving_projection_preserve_normal_direction",
            "strict_normal_boundary_update",
            default=False,
        )
    )
    normals = None
    if preserve_normal_direction:
        normals = _boundary_normal_unit_vectors(design_node_ids, cache, geometry_contract, mesh=mesh)
        if normals is None or normals.shape[0] != design_node_ids.size:
            preserve_normal_direction = False
    fixed_node_ids = _fixed_node_ids(cache, geometry_contract, options)
    fixed_mask = bm.isin(design_node_ids, fixed_node_ids)

    area_gradient = _polygon_area_gradient(node_coords)
    if preserve_normal_direction and normals is not None:
        scalar_area_gradient = bm.einsum("ij,ij->i", area_gradient[:, : normals.shape[1]], normals)
        denominator = float(bm.dot(scalar_area_gradient, scalar_area_gradient))
        if denominator <= 0.0:
            return raw_gradient

        if isinstance(raw_gradient, Mapping):
            dim = int(normals.shape[1])
            gradient = bm.zeros(design_node_ids.size, dtype=float)
            node_keys = bm.asarray(list(raw_gradient.keys()), dtype=int)
            node_values = bm.asarray([_coerce_vector(value, dim) for value in raw_gradient.values()], dtype=float)
            order = bm.argsort(node_keys)
            node_keys = node_keys[order]
            node_values = node_values[order]
            positions = bm.searchsorted(node_keys, design_node_ids)
            valid = positions < node_keys.size
            if bm.any(valid):
                matched = bm.zeros_like(valid)
                valid_positions = positions[valid]
                matched[valid] = node_keys[valid_positions] == design_node_ids[valid]
                valid &= matched
            if bm.any(valid):
                gradient[valid] = bm.einsum("ij,ij->i", node_values[positions[valid]], normals[valid])
            if fixed_mask.any():
                gradient[fixed_mask] = 0.0
            numerator = float(bm.dot(gradient, scalar_area_gradient))
            projected = gradient - (numerator / denominator) * scalar_area_gradient
            if fixed_mask.any():
                projected[fixed_mask] = 0.0
            return {int(node_id): tuple(float(component) for component in projected[index] * normals[index]) for index, node_id in enumerate(design_node_ids.tolist())}

        values = bm.asarray(raw_gradient, dtype=float)
        if values.ndim == 1 and values.size == design_node_ids.size:
            gradient = values.astype(float, copy=True)
            if fixed_mask.any():
                gradient[fixed_mask] = 0.0
            numerator = float(bm.dot(gradient, scalar_area_gradient))
            projected = gradient - (numerator / denominator) * scalar_area_gradient
            if fixed_mask.any():
                projected[fixed_mask] = 0.0
            return projected[:, None] * normals
        if values.ndim >= 2 and values.shape[0] == design_node_ids.size and values.shape[1] >= normals.shape[1]:
            gradient = bm.einsum("ij,ij->i", values[:, : normals.shape[1]], normals)
            if fixed_mask.any():
                gradient[fixed_mask] = 0.0
            numerator = float(bm.dot(gradient, scalar_area_gradient))
            projected = gradient - (numerator / denominator) * scalar_area_gradient
            if fixed_mask.any():
                projected[fixed_mask] = 0.0
            result = values.astype(float, copy=True)
            result[:, : normals.shape[1]] = projected[:, None] * normals
            return result
        return raw_gradient

    denominator = float(bm.sum(area_gradient * area_gradient))
    if denominator <= 0.0:
        return raw_gradient

    if isinstance(raw_gradient, Mapping):
        gradient = bm.zeros((design_node_ids.size, 2), dtype=float)
        node_keys = bm.asarray(list(raw_gradient.keys()), dtype=int)
        node_values = bm.asarray([_coerce_vector(value, 2) for value in raw_gradient.values()], dtype=float)
        order = bm.argsort(node_keys)
        node_keys = node_keys[order]
        node_values = node_values[order]
        positions = bm.searchsorted(node_keys, design_node_ids)
        valid = positions < node_keys.size
        if bm.any(valid):
            matched = bm.zeros_like(valid)
            valid_positions = positions[valid]
            matched[valid] = node_keys[valid_positions] == design_node_ids[valid]
            valid &= matched
        if bm.any(valid):
            gradient[valid] = node_values[positions[valid]]
        numerator = float(bm.sum(gradient * area_gradient))
        projected = gradient - (numerator / denominator) * area_gradient
        return {int(node_id): (float(vector[0]), float(vector[1])) for node_id, vector in zip(design_node_ids.tolist(), projected, strict=True)}

    values = bm.asarray(raw_gradient, dtype=float)
    if values.ndim == 2 and values.shape[0] == design_node_ids.size and values.shape[1] >= 2:
        gradient = values[:, :2].astype(float, copy=True)
        numerator = float(bm.sum(gradient * area_gradient))
        projected = gradient - (numerator / denominator) * area_gradient
        result = values.astype(float, copy=True)
        result[:, :2] = projected
        return result

    return raw_gradient
