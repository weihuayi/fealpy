"""目标函数评估骨架。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fealpy.functional import integral
from fealpy.backend import backend_manager as bm

from .benchmark_common import get_value


@dataclass(slots=True)
class ObjectiveEvaluationResult:
    """目标函数评估结果。"""

    dissipation: float | None
    volume_term: float | None
    barycenter_term: float | None
    regularization_term: float | None
    total_objective: float | None
    adjoint_rhs: Any
    contributions: dict[str, Any] = field(default_factory=dict)
    adjoint_rhs_source: Any = None
    adjoint_rhs_source_kind: str | None = None
    shape_derivative: Any = None
    shape_derivative_source: Any = None
    shape_derivative_source_kind: str | None = None


@dataclass(slots=True)
class ObjectiveDerivativeSource:
    """目标导数的可执行源项入口。"""

    source: Any
    description: str = "目标导数源项"
    kind: str = "objective_derivative_source"
    coordtype: str = "barycentric"

    def __call__(self, bcs: Any, index: Any = None) -> Any:
        """在给定积分点上生成源项值。"""
        if index is None:
            return self.source(bcs)
        return self.source(bcs, index=index)


def _build_ns_dissipation_rhs_from_state(
    velocity: Any,
    pressure: Any,
    objective_parameters: Any,
    scale: float = 2.0,
) -> Any:
    """基于稳态 NS 的耗散目标构造伴随右端。

    优先使用离散意义下与论文一致的严格 RHS:
    .. math::
        -\\alpha\\mu \\Delta u
    其中 ``alpha`` 对应 ``scale``。若当前速度场对象无法提供二阶导数，
    再回退到旧的连续代理拼法，避免影响其它未迁移的算例。
    """
    if velocity is None:
        return None

    viscosity = get_value(objective_parameters, "viscosity", "mu", default=1.0)
    rho = get_value(objective_parameters, "rho", default=1.0)
    body_force = get_value(objective_parameters, "body_force", "source", default=0.0)
    space = get_value(velocity, "space", default=None)
    scalar_space = get_value(space, "scalar_space", default=None)
    dof_priority = bool(get_value(space, "dof_priority", default=False))
    dof_numel = int(get_value(space, "dof_numel", default=0) or 0)
    if (
        space is not None
        and scalar_space is not None
        and hasattr(scalar_space, "hess_basis")
        and hasattr(space, "cell_to_dof")
        and dof_numel > 0
    ):
        try:
            scalar_ldof = int(scalar_space.number_of_local_dofs("cell"))
            scalar_hess_basis = scalar_space.hess_basis
            velocity_values = bm.asarray(velocity)

            def source(bcs: Any, index: Any = None) -> Any:
                hessian = scalar_hess_basis(bcs, index=index)
                if hessian is None:
                    return None
                cell_dofs = space.cell_to_dof(index=index)
                if dof_priority:
                    cell_dofs = bm.reshape(cell_dofs, (-1, dof_numel, scalar_ldof))
                else:
                    cell_dofs = bm.swapaxes(bm.reshape(cell_dofs, (-1, scalar_ldof, dof_numel)), 1, 2)
                trace_hessian = bm.einsum("...ii->...", hessian)
                component_values = velocity_values[cell_dofs]
                laplace_u = bm.einsum("cqk,cdk->cqd", trace_hessian, component_values)
                return -float(scale) * float(viscosity) * laplace_u

            return ObjectiveDerivativeSource(source=source)
        except Exception:
            pass

    def source(bcs: Any, index: Any = None) -> Any:
        u = velocity(bcs) if index is None else velocity(bcs, index=index)
        grad_u = velocity.grad_value(bcs) if index is None else velocity.grad_value(bcs, index=index)
        convective = bm.einsum("...j,...ij->...i", u, grad_u)
        result = -float(scale) * float(rho) * convective

        if pressure is not None and hasattr(pressure, "grad_value"):
            grad_p = pressure.grad_value(bcs) if index is None else pressure.grad_value(bcs, index=index)
            result -= float(scale) * grad_p

        if callable(body_force):
            load = body_force(bcs) if index is None else body_force(bcs, index=index)
            result += float(scale) * load
        elif body_force not in (None, 0, 0.0):
            result += float(scale) * float(body_force)

        return result

    return ObjectiveDerivativeSource(source=source)


def _evaluate_dissipation_from_velocity(mesh: Any, velocity: Any, objective_parameters: Any) -> float | None:
    """基于速度场进行粘性耗散积分。"""
    if mesh is None or velocity is None:
        return None
    if not hasattr(velocity, "grad_value"):
        return None
    q = get_value(objective_parameters, "q", "quadrature_order", default=None)
    if q is None:
        space = get_value(velocity, "space")
        if space is not None:
            q = int(get_value(space, "p", default=1)) + 2
        else:
            q = 3
    viscosity = get_value(objective_parameters, "viscosity", "mu", default=1.0)
    qf = mesh.quadrature_formula(q, etype="cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    grad_u = velocity.grad_value(bcs)
    energy_density = bm.sum(grad_u * grad_u, axis=(-1, -2))
    cell_measure = mesh.entity_measure("cell")
    value = integral(energy_density, ws, cell_measure)
    return float(viscosity) * float(value)


def calculate_dissipation_objective(
    mesh: Any,
    state_result: Any,
    objective_parameters: Any,
    current_state: Any = None,
) -> Any:
    """计算耗散目标。"""
    value = get_value(state_result, "dissipation", "dissipation_term")
    if value is not None:
        return value
    value = get_value(objective_parameters, "dissipation_objective", "dissipation_term")
    if value is not None:
        return value(mesh, state_result, objective_parameters, current_state) if callable(value) else value

    mesh = mesh if mesh is not None else get_value(state_result, "mesh")
    if mesh is None:
        mesh = get_value(current_state, "mesh")
    velocity = get_value(state_result, "velocity", "state_velocity", "u")
    if velocity is None:
        velocity = get_value(current_state, "velocity", "state_velocity", "u")
    return _evaluate_dissipation_from_velocity(mesh, velocity, objective_parameters)


def calculate_total_objective(
    dissipation: Any,
    volume_term: Any,
    barycenter_term: Any,
    regularization_term: Any,
) -> Any:
    """计算总目标。"""
    return float(
        sum(
            float(term)
            for term in (dissipation, volume_term, barycenter_term, regularization_term)
            if term is not None
        )
    )


def calculate_adjoint_rhs(
    mesh: Any,
    state_result: Any,
    objective_parameters: Any,
    current_state: Any = None,
) -> Any:
    """计算伴随右端。"""
    value = get_value(objective_parameters, "adjoint_rhs", "adjoint_rhs_builder")
    if value is not None:
        value = value(mesh, state_result, objective_parameters, current_state) if callable(value) else value
        if isinstance(value, ObjectiveDerivativeSource):
            return value
        if callable(value):
            return ObjectiveDerivativeSource(source=value)
        return value

    velocity = get_value(state_result, "velocity", "state_velocity", "u")
    if velocity is None:
        velocity = get_value(current_state, "velocity", "state_velocity", "u")
    pressure = get_value(state_result, "pressure", "state_pressure", "p")
    scale = get_value(objective_parameters, "adjoint_rhs_scale", "dissipation_rhs_scale", default=2.0)
    if velocity is None:
        return None
    space = get_value(velocity, "space", default=None)
    scalar_space = get_value(space, "scalar_space", default=None)
    if space is not None and scalar_space is not None:
        return _build_ns_dissipation_rhs_from_state(velocity, pressure, objective_parameters, float(scale))

    def source(bcs: Any, index: Any = None) -> Any:
        return float(scale) * (velocity(bcs) if index is None else velocity(bcs, index=index))

    return ObjectiveDerivativeSource(source=source)


def calculate_shape_derivative(
    mesh: Any,
    state_result: Any,
    objective_parameters: Any,
    current_state: Any = None,
    *,
    contributions: Mapping[str, Any] | None = None,
    total_objective: Any = None,
) -> Any:
    """计算形状导数源项。"""
    value = get_value(objective_parameters, "shape_derivative", "shape_derivative_builder", default=None)
    if callable(value):
        value = value(
            mesh,
            state_result,
            objective_parameters,
            current_state,
            contributions=contributions,
            total_objective=total_objective,
        )

    source = get_value(objective_parameters, "shape_derivative_source", default=None)
    if callable(source):
        source = source(
            mesh,
            state_result,
            objective_parameters,
            current_state,
            contributions=contributions,
            total_objective=total_objective,
        )
    if source is not None:
        value = source

    regularization = get_value(objective_parameters, "geometry_regularization", "geometric_regularization", default=None)
    if regularization is not None:
        source = get_value(regularization, "shape_derivative_source", default=None)
        if source is not None:
            if callable(source):
                source = source(
                    mesh,
                    state_result,
                    objective_parameters,
                    current_state,
                    contributions=contributions,
                    total_objective=total_objective,
                )
            if source is not None:
                if value is None:
                    value = source
                elif isinstance(value, Mapping) and isinstance(source, Mapping):
                    value = {
                        int(key): tuple(
                            bm.asarray(value.get(key, 0.0), dtype=float).reshape(-1)
                            + bm.asarray(source.get(key, 0.0), dtype=float).reshape(-1)
                        )
                        for key in set(value) | set(source)
                    }
    return value


def update_geometric_quantities(
    mesh: Any,
    state_result: Any,
    objective_parameters: Any,
    current_state: Any = None,
) -> None:
    """Refresh cashocs-style geometric regularization state before evaluation."""
    regularization = get_value(objective_parameters, "geometry_regularization", "geometric_regularization", default=None)
    if regularization is None:
        return
    updater = get_value(regularization, "update_geometric_quantities", "update", default=None)
    if not callable(updater):
        return
    try:
        updater(mesh, objective_parameters=objective_parameters, current_state=current_state)
    except TypeError:
        updater(mesh, state_result, objective_parameters, current_state)


def evaluate_objective(
    mesh: Any,
    state_result: Any,
    objective_parameters: Any,
    current_state: Any = None,
) -> ObjectiveEvaluationResult:
    """评估目标函数。"""
    update_geometric_quantities(mesh, state_result, objective_parameters, current_state)
    dissipation = calculate_dissipation_objective(mesh, state_result, objective_parameters, current_state)
    volume_term = get_value(objective_parameters, "volume_term", "volume_penalty")
    volume_term = volume_term(mesh, state_result, objective_parameters, current_state) if callable(volume_term) else volume_term
    barycenter_term = get_value(objective_parameters, "barycenter_term", "barycenter_penalty")
    barycenter_term = barycenter_term(mesh, state_result, objective_parameters, current_state) if callable(barycenter_term) else barycenter_term
    regularization_term = get_value(objective_parameters, "regularization_term", "regularization")
    regularization_term = regularization_term(mesh, state_result, objective_parameters, current_state) if callable(regularization_term) else regularization_term
    total_objective = float(
        sum(
            float(term)
            for term in (dissipation, volume_term, barycenter_term, regularization_term)
            if term is not None
        )
    )
    adjoint_rhs = calculate_adjoint_rhs(mesh, state_result, objective_parameters, current_state)
    contributions = {
        "dissipation": dissipation,
        "volume_term": volume_term,
        "barycenter_term": barycenter_term,
        "regularization_term": regularization_term,
    }
    adjoint_rhs_source = adjoint_rhs
    adjoint_rhs_source_kind = getattr(adjoint_rhs_source, "kind", None)
    shape_derivative = calculate_shape_derivative(
        mesh,
        state_result,
        objective_parameters,
        current_state,
        contributions=contributions,
        total_objective=total_objective,
    )
    shape_derivative_source = shape_derivative
    shape_derivative_source_kind = getattr(shape_derivative_source, "kind", None)

    return ObjectiveEvaluationResult(
        dissipation=dissipation,
        volume_term=volume_term,
        barycenter_term=barycenter_term,
        regularization_term=regularization_term,
        total_objective=total_objective,
        adjoint_rhs=adjoint_rhs,
        contributions=contributions,
        adjoint_rhs_source=adjoint_rhs_source,
        adjoint_rhs_source_kind=adjoint_rhs_source_kind,
        shape_derivative=shape_derivative,
        shape_derivative_source=shape_derivative_source,
        shape_derivative_source_kind=shape_derivative_source_kind,
    )
