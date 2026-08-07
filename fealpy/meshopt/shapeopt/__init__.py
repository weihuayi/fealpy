"""Reference materials for reproducing cashocs-style shape optimization in FEALPy."""

from fealpy.model.stokes.exp0009 import ObstacleStokesFluidModel

from .adjoint_solver import AdjointSolveResult, solve_adjoint_system
from .benchmark_runner import ShapeOptimizationRunner
from .geometry_contract import (
    GeometryContract,
    OptimizationCache,
    build_geometry_contract,
    extract_design_nodes,
    extract_fixed_nodes,
    extract_free_nodes,
    initialize_optimization_cache,
    refresh_propagation_cache,
)
from .geometry_gradient import (
    GeometryGradientResult,
    assemble_geometry_gradient,
    build_boundary_displacement,
    build_trial_update,
    compute_descent_direction,
    filter_fixed_boundaries,
    project_to_boundary_normals,
    project_to_area_preserving_tangent_space,
)
from .geometry_regularization import ObstacleGeometryRegularization
from .mesh_propagation import RemeshResult as MeshRemeshResult, TrialState, check_mesh_quality, propagate_mesh, remesh_mesh
from .objective import (
    ObjectiveEvaluationResult,
    ObjectiveDerivativeSource,
    calculate_adjoint_rhs,
    calculate_dissipation_objective,
    calculate_total_objective,
    evaluate_objective,
)
from .shape_optimizer import (
    OptimizationHistoryEntry,
    OptimizationResult,
    LineSearchResult,
    RemeshResult,
    OneStepOptimizationCheckResult,
    StepResult,
    ShapeOptimizer,
    check_one_step_optimization_finite_difference,
)
from .l_bfgs import ShapeLBFGSState
from .state_solver import StateSolveResult, solve_state_system

__all__ = [
    "AdjointSolveResult",
    "ObstacleStokesFluidModel",
    "GeometryContract",
    "OptimizationCache",
    "GeometryGradientResult",
    "ObstacleGeometryRegularization",
    "OptimizationHistoryEntry",
    "OptimizationResult",
    "LineSearchResult",
    "ShapeLBFGSState",
    "RemeshResult",
    "OneStepOptimizationCheckResult",
    "StepResult",
    "ObjectiveEvaluationResult",
    "ObjectiveDerivativeSource",
    "ShapeOptimizer",
    "ShapeOptimizationRunner",
    "StateSolveResult",
    "TrialState",
    "MeshRemeshResult",
    "assemble_geometry_gradient",
    "build_boundary_displacement",
    "build_geometry_contract",
    "build_trial_update",
    "check_one_step_optimization_finite_difference",
    "calculate_adjoint_rhs",
    "calculate_dissipation_objective",
    "calculate_total_objective",
    "check_mesh_quality",
    "compute_descent_direction",
    "evaluate_objective",
    "assemble_adjoint_weak_form",
    "extract_design_nodes",
    "extract_fixed_nodes",
    "extract_free_nodes",
    "filter_fixed_boundaries",
    "initialize_optimization_cache",
    "project_to_area_preserving_tangent_space",
    "project_to_boundary_normals",
    "propagate_mesh",
    "remesh_mesh",
    "refresh_propagation_cache",
    "solve_adjoint_system",
    "solve_state_system",
]
