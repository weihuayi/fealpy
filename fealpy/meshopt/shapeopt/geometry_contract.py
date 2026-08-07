"""几何合同与缓存骨架。

这一层只负责几何、边界分组和缓存，不负责状态方程和优化迭代。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from fealpy.backend import backend_manager as bm


BoundaryIds = tuple[int, ...]


def _as_boundary_ids(value: Any) -> BoundaryIds:
    """把任意节点集合规整成稳定的整数元组。"""
    if value is None:
        return ()
    array = bm.asarray(value, dtype=int).reshape(-1)
    if array.size == 0:
        return ()
    return tuple(int(node) for node in bm.unique(array).tolist())


def _mesh_boundary_ids(mesh: Any, *names: str) -> BoundaryIds:
    """从 mesh 属性中提取节点集合，按给定名字依次回退。"""
    collected = [boundary_id for name in names for boundary_id in _as_boundary_ids(getattr(mesh, name, None))]
    if not collected:
        return ()
    return tuple(int(node) for node in bm.unique(bm.asarray(collected, dtype=int)).tolist())


@dataclass(slots=True)
class GeometryContract:
    """第一版形状优化所需的几何合同。"""

    spatial_dim: int
    inlet_boundary_ids: BoundaryIds = ()
    outlet_boundary_ids: BoundaryIds = ()
    fixed_boundary_ids: BoundaryIds = ()
    design_boundary_ids: BoundaryIds = ()
    node_classification: dict[int, str] = field(default_factory=dict)
    boundary_node_order: BoundaryIds = ()


@dataclass(slots=True)
class OptimizationCache:
    """第一版形状优化所需的缓存。"""

    reference_mesh: Any
    mesh_revision: int = 0
    fixed_nodes: BoundaryIds = ()
    design_nodes: BoundaryIds = ()
    free_nodes: BoundaryIds = ()
    normal_info: Any = None
    propagation_parameters: dict[str, Any] = field(default_factory=dict)
    reference_objective: float | None = None
    state_solution_valid: bool = False
    adjoint_solution_valid: bool = False
    gradient_solution_valid: bool = False
    scalar_product_valid: bool = False


def build_geometry_contract(
    mesh: Any,
    boundary_markers: Mapping[str, Sequence[int]],
    spatial_dim: int | None = None,
) -> GeometryContract:
    """根据网格和边界标记构造几何合同。"""
    if spatial_dim is None:
        spatial_dim = getattr(mesh, "geo_dimension", None)
        if spatial_dim is None:
            spatial_dim = getattr(mesh, "topological_dimension", None)
        if spatial_dim is None:
            spatial_dim = 2

    inlet_boundary_ids = _as_boundary_ids(boundary_markers.get("inlet"))
    if not inlet_boundary_ids:
        inlet_boundary_ids = _mesh_boundary_ids(mesh, "inlet_nodes", "inlet_boundary_ids", "inlet")

    outlet_boundary_ids = _as_boundary_ids(boundary_markers.get("outlet"))
    if not outlet_boundary_ids:
        outlet_boundary_ids = _mesh_boundary_ids(mesh, "outlet_nodes", "outlet_boundary_ids", "outlet")

    fixed_boundary_ids = _as_boundary_ids(boundary_markers.get("fixed"))
    if not fixed_boundary_ids:
        fixed_boundary_ids = _mesh_boundary_ids(
            mesh,
            "fixed_nodes",
            "fixed_boundary_ids",
            "shape_bdry_fix",
            "fixed_left_nodes",
            "fixed_right_nodes",
        )

    design_boundary_ids = _as_boundary_ids(boundary_markers.get("design"))
    if not design_boundary_ids:
        design_boundary_ids = _mesh_boundary_ids(
            mesh,
            "design_nodes",
            "design_boundary_ids",
            "shape_bdry_def",
        )

    node_classification = (
        {boundary_id: "inlet" for boundary_id in inlet_boundary_ids}
        | {boundary_id: "outlet" for boundary_id in outlet_boundary_ids}
        | {boundary_id: "fixed" for boundary_id in fixed_boundary_ids}
        | {boundary_id: "design" for boundary_id in design_boundary_ids}
    )

    boundary_node_order = tuple(
        dict.fromkeys(
            inlet_boundary_ids
            + outlet_boundary_ids
            + fixed_boundary_ids
            + design_boundary_ids
        )
    )

    return GeometryContract(
        spatial_dim=spatial_dim,
        inlet_boundary_ids=inlet_boundary_ids,
        outlet_boundary_ids=outlet_boundary_ids,
        fixed_boundary_ids=fixed_boundary_ids,
        design_boundary_ids=design_boundary_ids,
        node_classification=node_classification,
        boundary_node_order=boundary_node_order,
    )


def initialize_optimization_cache(
    reference_mesh: Any,
    geometry_contract: GeometryContract,
    propagation_parameters: Mapping[str, Any] | None = None,
    reference_objective: float | None = None,
) -> OptimizationCache:
    """根据几何合同初始化缓存。"""
    propagation_parameters = dict(propagation_parameters or {})
    return OptimizationCache(
        reference_mesh=reference_mesh,
        mesh_revision=0,
        fixed_nodes=extract_fixed_nodes(reference_mesh, geometry_contract),
        design_nodes=extract_design_nodes(reference_mesh, geometry_contract),
        free_nodes=extract_free_nodes(reference_mesh, geometry_contract),
        normal_info=None,
        propagation_parameters=propagation_parameters,
        reference_objective=reference_objective,
        state_solution_valid=False,
        adjoint_solution_valid=False,
        gradient_solution_valid=False,
        scalar_product_valid=False,
    )


def refresh_propagation_cache(
    cache: OptimizationCache,
    accepted_state: Any,
    accepted_objective: float | None = None,
    *,
    invalidate_solution_cache: bool = True,
    clear_reference_objective: bool = False,
) -> OptimizationCache:
    """把接受后的试探状态写回缓存。"""
    if isinstance(accepted_state, Mapping):
        reference_mesh = accepted_state.get("mesh", cache.reference_mesh)
    else:
        reference_mesh = getattr(accepted_state, "mesh", cache.reference_mesh)

    mesh_revision = cache.mesh_revision + 1 if reference_mesh is not cache.reference_mesh else cache.mesh_revision
    if invalidate_solution_cache:
        state_solution_valid = False
        adjoint_solution_valid = False
        gradient_solution_valid = False
        scalar_product_valid = False
    else:
        state_solution_valid = cache.state_solution_valid
        adjoint_solution_valid = cache.adjoint_solution_valid
        gradient_solution_valid = cache.gradient_solution_valid
        scalar_product_valid = cache.scalar_product_valid

    if clear_reference_objective:
        reference_objective = None
    else:
        reference_objective = accepted_objective if accepted_objective is not None else cache.reference_objective

    return OptimizationCache(
        reference_mesh=reference_mesh,
        mesh_revision=mesh_revision,
        fixed_nodes=cache.fixed_nodes,
        design_nodes=cache.design_nodes,
        free_nodes=cache.free_nodes,
        normal_info=cache.normal_info,
        propagation_parameters=dict(cache.propagation_parameters),
        reference_objective=reference_objective,
        state_solution_valid=state_solution_valid,
        adjoint_solution_valid=adjoint_solution_valid,
        gradient_solution_valid=gradient_solution_valid,
        scalar_product_valid=scalar_product_valid,
    )


def extract_design_nodes(
    mesh: Any,
    geometry_contract: GeometryContract,
) -> BoundaryIds:
    """提取设计边界节点。"""
    mesh_nodes = _mesh_boundary_ids(mesh, "design_nodes", "design_boundary_ids", "shape_bdry_def")
    if mesh_nodes:
        return mesh_nodes
    return geometry_contract.design_boundary_ids


def extract_fixed_nodes(
    mesh: Any,
    geometry_contract: GeometryContract,
) -> BoundaryIds:
    """提取固定边界节点。"""
    mesh_nodes = _mesh_boundary_ids(
        mesh,
        "fixed_nodes",
        "fixed_boundary_ids",
        "shape_bdry_fix",
        "fixed_left_nodes",
        "fixed_right_nodes",
    )
    if mesh_nodes:
        return mesh_nodes
    return geometry_contract.fixed_boundary_ids


def extract_free_nodes(
    mesh: Any,
    geometry_contract: GeometryContract,
) -> BoundaryIds:
    """提取自由节点。"""
    mesh_nodes = _mesh_boundary_ids(mesh, "free_nodes")
    if mesh_nodes:
        return mesh_nodes
    return tuple(boundary_id for boundary_id, label in geometry_contract.node_classification.items() if label not in {"fixed", "design", "inlet", "outlet"})
