"""伴随方程求解器骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fealpy.backend import backend_manager as bm
from fealpy.fem import (
    DirichletBC,
    LinearBlockForm,
    LinearForm,
    VectorSourceIntegrator,
)
from fealpy.solver import spsolve

from .geometry_regularization import (
        _polygon_vertex_normals,
        _polygon_vertex_weights,
)
from .objective import ObjectiveDerivativeSource


@dataclass(slots=True)
class AdjointSolveResult:
    """伴随方程求解结果。"""

    adjoint: Any = None
    system_matrix: Any = None
    rhs: Any = None
    adjoint_vector: Any = None
    adjoint_rhs: Any = None
    mesh: Any = None
    shape_derivative: Any = None
    shape_density: Any = None


@dataclass(slots=True)
class AdjointWeakForm:
    """FEALPy 伴随弱式打包结果。"""

    mesh: Any
    velocity_space: Any
    pressure_space: Any
    matrix: Any
    load_form: Any
    linear_block_form: Any
    adjoint_rhs_source: Any = None
    boundary_conditions: tuple[Any, ...] = ()


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    """从映射或对象属性里取第一个可用字段。"""
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _velocity_gradient_at_nodes(mesh: Any, field: Any) -> Any | None:
    if field is None or not hasattr(field, "space") or not hasattr(field.space, "grad_value"):
        return None
    try:
        qf = mesh.quadrature_formula(q=4, etype="cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad_u = field.space.grad_value(uh=field, bc=bcs)
        grad_u = bm.einsum("n, knij -> kij", ws, grad_u)
        cellmeasure = mesh.entity_measure("cell")
        n2c = mesh.node_to_cell()
        weights = bm.ones(n2c.shape)
        weights *= cellmeasure
        weights = n2c.mul(weights)
        weights = weights.toarray()
        weights_sum = bm.sum(weights, axis=1)
        weights = weights / weights_sum[:, None]
        grad_at_nodes = bm.einsum("lk, kij -> lij", weights, grad_u)
        return bm.asarray(grad_at_nodes, dtype=float)
    except Exception:
        return None


def _ns_boundary_shape_derivative(
    mesh: Any,
    state_result: Any,
    adjoint_result: Any,
    geometry_contract: Any,
    options: Any = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]] | None:
    """构造稳态 NS 耗散目标的边界形状导数代表。"""
    state_velocity = get_value(state_result, "velocity", "state_velocity", "u")
    if state_velocity is None:
        return None

    objective_parameters = get_value(options, "objective_parameters", default=None)
    is_hole_boundary = get_value(objective_parameters, "is_hole_boundary", default=None)
    resolved_objective_parameters = objective_parameters if isinstance(objective_parameters, Mapping) else {}
    design_node_order = get_value(
        resolved_objective_parameters,
        "design_boundary_node_order",
        "design_boundary_node_ids",
        default=None,
    )
    if design_node_order is None:
        return None

    design_node_ids = bm.asarray(design_node_order, dtype=int).reshape(-1)
    if design_node_ids.size == 0:
        return None

    if isinstance(mesh, Mapping):
        nodes = mesh.get("node", mesh.get("nodes"))
    else:
        nodes = get_value(mesh, "node", "nodes")
    if nodes is None:
        return None
    nodes = bm.asarray(nodes, dtype=float)

    coords = nodes[design_node_ids]
    normals = _polygon_vertex_normals(coords, is_hole_boundary=is_hole_boundary)
    vertex_weights = _polygon_vertex_weights(coords)
    state_grad_at_nodes = _velocity_gradient_at_nodes(mesh, state_velocity)
    if state_grad_at_nodes is None:
        return None

    viscosity = float(
        resolved_objective_parameters.get(
            "viscosity",
            get_value(state_result, "viscosity", "mu", "nu", default=1.0),
        )
    )
    num_design_nodes = int(design_node_ids.size)
    # 收集状态速度梯度：shape = (N_gamma, 2, 2)
    selected_state_grad = bm.zeros((num_design_nodes, 2, 2), dtype=float)

    valid_state_nodes = (
        (design_node_ids >= 0)
        & (design_node_ids < int(state_grad_at_nodes.shape[0]))
    )
    if bm.any(valid_state_nodes):
        valid_ids = design_node_ids[valid_state_nodes]
        selected_state_grad[valid_state_nodes] = bm.asarray(
            state_grad_at_nodes[valid_ids],
            dtype=float,
        )
    # state_normal[i] = grad_u(x_i) @ n_i
    # shape = (N_gamma, 2)
    state_normal = bm.einsum(
        "nij,nj->ni",
        selected_state_grad,
        normals,
    )
    density_values = -float(viscosity) * bm.einsum(
        "ni,ni->n",
        state_normal,
        state_normal,
    )
    adjoint = get_value(adjoint_result, "adjoint", default=None)
    adjoint_velocity = get_value(
        adjoint,
        "velocity",
        "state_velocity",
        "u",
        default=None,
    )
    adjoint_grad_at_nodes = (
        _velocity_gradient_at_nodes(mesh, adjoint_velocity)
        if adjoint_velocity is not None
        else None
    )
    if adjoint_grad_at_nodes is not None:
        selected_adjoint_grad = bm.zeros((num_design_nodes, 2, 2), dtype=float)
        valid_adjoint_nodes = (
            (design_node_ids >= 0)
            & (design_node_ids < int(adjoint_grad_at_nodes.shape[0]))
        )
        if bm.any(valid_adjoint_nodes):
            valid_ids = design_node_ids[valid_adjoint_nodes]
            selected_adjoint_grad[valid_adjoint_nodes] = bm.asarray(
                adjoint_grad_at_nodes[valid_ids],
                dtype=float,
            )
        # adjoint_normal[i] = grad_lambda(x_i) @ n_i
        # shape = (N_gamma, 2)
        adjoint_normal = bm.einsum(
            "nij,nj->ni",
            selected_adjoint_grad,
            normals,
        )
        adjoint_correction = float(viscosity) * bm.einsum(
            "ni,ni->n",
            adjoint_normal,
            state_normal,
        )
        density_values = density_values - adjoint_correction
    # nodal vector representative:
    # gradient_i = density_i * n_i * vertex_weight_i
    gradient = (
        density_values[:, None]
        * bm.asarray(normals, dtype=float)
        * bm.asarray(vertex_weights, dtype=float)[:, None]
    )
    return {
        int(node_id): (float(vector[0]), float(vector[1]))
        for node_id, vector in zip(design_node_ids.tolist(), gradient, strict=True)
    }, {
        int(node_id): float(density)
        for node_id, density in zip(design_node_ids.tolist(), density_values, strict=True)
    }


def _normalize_adjoint_rhs(adjoint_rhs: Any) -> tuple[Any, Any]:
    """把目标导数源项规范成可装配对象和元数据。"""
    if isinstance(adjoint_rhs, ObjectiveDerivativeSource):
        return adjoint_rhs.source, adjoint_rhs
    if callable(adjoint_rhs):
        wrapped = ObjectiveDerivativeSource(source=adjoint_rhs)
        return wrapped.source, wrapped
    return adjoint_rhs, adjoint_rhs


def _build_adjoint_dirichlet_bc(
    state_result: Any,
    velocity_space: Any,
    pressure_space: Any,
    options: Any = None,
) -> DirichletBC:
    """构造伴随速度/压力边界条件。"""
    velocity_gd = get_value(
        state_result,
        "adjoint_velocity_dirichlet_data",
        "adjoint_inlet_velocity",
        "adjoint_velocity_bc",
        default=0.0,
    )
    velocity_threshold = get_value(
        state_result,
        "adjoint_velocity_dirichlet_threshold",
        "velocity_dirichlet_threshold",
        "adjoint_velocity_boundary_threshold",
        "adjoint_velocity_threshold",
        default=velocity_space.is_boundary_dof(),
    )
    pressure_gd = get_value(state_result, "adjoint_pressure_dirichlet_data", "adjoint_pressure_bc", default=0.0)
    if isinstance(pressure_gd, (int, float)):
        pressure_gd = bm.zeros(
            pressure_space.number_of_global_dofs(),
            dtype=float,
        ) + pressure_gd
    pressure_threshold = get_value(
        state_result,
        "adjoint_pressure_dirichlet_threshold",
        "pressure_dirichlet_threshold",
        "adjoint_pressure_boundary_threshold",
        default=bm.zeros(
            pressure_space.number_of_global_dofs(),
            dtype=bool,
        ),
    )

    return DirichletBC(
        (velocity_space, pressure_space),
        gd=(velocity_gd, pressure_gd),
        threshold=(velocity_threshold, pressure_threshold),
        method=get_value(options, "boundary_method", default="interp"),
    )


def assemble_adjoint_weak_form(
    mesh: Any,
    state_result: Any,
    adjoint_rhs: Any,
    geometry_contract: Any,
    options: Any = None,
) -> AdjointWeakForm:
    spaces = (
        get_value(state_result, "velocity_space", "uspace", "u_space"),
        get_value(state_result, "pressure_space", "pspace", "p_space"),
    )
    velocity_space, pressure_space = spaces[0], spaces[1]
    state_matrix = get_value(state_result, "system_matrix", "matrix", default=None)
    matrix = state_matrix.T
    
    source, source_metadata = _normalize_adjoint_rhs(adjoint_rhs)
    q = get_value(options, "q", default=None)
    load_form = LinearForm(velocity_space)
    if source not in (None, 0, 0.0):
        load_form.add_integrator(VectorSourceIntegrator(source, q=q))
    return AdjointWeakForm(
        mesh=mesh,
        velocity_space=velocity_space,
        pressure_space=pressure_space,
        matrix=matrix,
        load_form=load_form,
        linear_block_form=LinearBlockForm([load_form, LinearForm(pressure_space)]),
        adjoint_rhs_source=source_metadata,
        boundary_conditions=(
            _build_adjoint_dirichlet_bc(state_result, velocity_space, pressure_space, options),
        ),
    )


def solve_adjoint_system(
    mesh: Any,
    state_result: Any,
    adjoint_rhs: Any,
    geometry_contract: Any,
    options: Any = None,
) -> AdjointSolveResult:
    """求解伴随方程。"""
    weak_form = assemble_adjoint_weak_form(mesh, state_result, adjoint_rhs, geometry_contract, options=options)
    # if isinstance(weak_form, AdjointWeakForm):
    velocity_space = weak_form.velocity_space
    pressure_space = weak_form.pressure_space
    
    velocity_dofs = velocity_space.number_of_global_dofs()
    pressure_dofs = pressure_space.number_of_global_dofs()
    mixed_dofs = int(velocity_dofs + pressure_dofs)
    matrix = weak_form.matrix
    state_matrix_shape = getattr(matrix, "shape", None)
    augmented_system = False
    if state_matrix_shape is not None:
        matrix_dims = tuple(int(value) for value in state_matrix_shape[:2])
        augmented_system = matrix_dims == (mixed_dofs + 1, mixed_dofs + 1)
         
    rhs = weak_form.linear_block_form.assembly(format="dense")
    if augmented_system:
        rhs = bm.concat([rhs, bm.zeros((1,), dtype=float)])
    bc = _build_adjoint_dirichlet_bc(state_result, velocity_space, pressure_space, options)

    boundary_dof_index = getattr(bc, "boundary_dof_index", None)
    if boundary_dof_index is not None and len(boundary_dof_index) > 0:
        rhs = bm.set_at(rhs, boundary_dof_index, 0.0)

    x = spsolve(matrix, rhs, solver=get_value(options, "linear_solver", default="mumps"))
    velocity_adjoin = velocity_space.function()
    pressure_adjoin = pressure_space.function()
    velocity_adjoin[:] = x[:velocity_dofs]
    if augmented_system:
        pressure_adjoin[:] = x[velocity_dofs:-1]
    else:
        pressure_adjoin[:] = x[velocity_dofs:velocity_dofs + pressure_dofs]
    result = AdjointSolveResult(
        adjoint={"velocity": velocity_adjoin, "pressure": pressure_adjoin},
        system_matrix=matrix,
        rhs=rhs,
        adjoint_vector=x,
        adjoint_rhs=adjoint_rhs,
        mesh=mesh,
    )
    if result.shape_derivative is None:
        ns_shape = _ns_boundary_shape_derivative(mesh, state_result, result, geometry_contract, options)
        if isinstance(ns_shape, tuple) and len(ns_shape) == 2:
            result.shape_derivative, result.shape_density = ns_shape
        else:
            result.shape_derivative = ns_shape

    return result
