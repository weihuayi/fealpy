"""Direct obstacle-in-Stokes benchmark script."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_stokes_lfem_model import StationaryIncompressibleStokesLFEMModel
from fealpy.mesher import BoxWithCircularHoleMesher2D
from fealpy.model.stokes.exp0009 import ObstacleStokesFluidModel

from fealpy.meshopt.shapeopt.benchmark_common import _remesh_quality_parameters
from fealpy.meshopt.shapeopt.benchmark_runner import ShapeOptimizationRunner
from fealpy.meshopt.shapeopt.geometry_regularization import ObstacleGeometryRegularization

DEFAULT_BOX = (-3.0, 3.0, -2.0, 2.0)
DEFAULT_CENTER = (0.0, 0.0)
DEFAULT_RADIUS = 0.5
DEFAULT_MESH_SIZE = 0.375
DEFAULT_MESH_SIZE_PROFILE = "graded"
DEFAULT_MESH_SIZE_INNER = 0.012
DEFAULT_MESH_SIZE_OUTER = 0.2
DEFAULT_MESH_SIZE_TRANSITION = 1.5
DEFAULT_INITIAL_STEP_SIZE = 0.03
DEFAULT_MAX_ITERATIONS = 800
DEFAULT_ALGORITHM = "lbfgs"  # "lbfgs" or "gradient_descent"
DEFAULT_RTOL = 5.0e-4
DEFAULT_LINEAR_SOLVER = "mumps"
DEFAULT_FACTOR_VOLUME = 1e3
DEFAULT_FACTOR_BARYCENTER = 0
DEFAULT_MU_DEF = 5e4
DEFAULT_MU_FIX = 1.0
DEFAULT_VISCOSITY = 1.0
DEFAULT_REMESH_QUALITY_PROFILE = "cashocs"
DEFAULT_BOUNDARY_MARKERS = {
    "inlet": (1,),
    "wall": (2,),
    "outlet": (3,),
    "fixed": (1, 2, 3),
    "design": (4,),
}
DEFAULT_OPTIONS = {
    "q": 3,
    "algorithm": DEFAULT_ALGORITHM,
    "rtol": DEFAULT_RTOL,
    "initial_step_size": DEFAULT_INITIAL_STEP_SIZE,
    "linear_solver": DEFAULT_LINEAR_SOLVER,
    "line_search_reduction": 0.5,
    "epsilon_armijo": 1.0e-4,
    "minimum_step_size": 1.0e-8,
    "max_line_search_iterations": 20,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "save_results": False,
    "save_txt": False,
    "state_system_is_linear": True,
}
DEFAULT_STATE_OPTIONS = {
    "run": "one_step",
    "solve": DEFAULT_LINEAR_SOLVER,
    "method": "Newton",
    "apply_bc": "dirichlet_dof",
    "pressure_gauge": "pin_dof",
    "pressure_gauge_dof": 0,
    "pressure_gauge_value": 0.0,
}

if __name__ == "__main__":  # pragma: no cover
    box = DEFAULT_BOX
    center = DEFAULT_CENTER
    radius = DEFAULT_RADIUS
    h = DEFAULT_MESH_SIZE
    mesh_size_profile = DEFAULT_MESH_SIZE_PROFILE
    mesh_size_inner = DEFAULT_MESH_SIZE_INNER
    mesh_size_outer = DEFAULT_MESH_SIZE_OUTER
    mesh_size_transition = DEFAULT_MESH_SIZE_TRANSITION
    viscosity = DEFAULT_VISCOSITY
    factor_volume = DEFAULT_FACTOR_VOLUME
    factor_barycenter = DEFAULT_FACTOR_BARYCENTER
    mu_def = DEFAULT_MU_DEF
    mu_fix = DEFAULT_MU_FIX
    algorithm = DEFAULT_ALGORITHM
    rtol = DEFAULT_RTOL
    initial_step_size = DEFAULT_INITIAL_STEP_SIZE
    max_iterations = DEFAULT_MAX_ITERATIONS
    remesh_quality_profile = DEFAULT_REMESH_QUALITY_PROFILE

    mesher = BoxWithCircularHoleMesher2D(
        {
            "box": box,
            "center": center,
            "radius": radius,
            "h": h,
            "mesh_size_profile": mesh_size_profile,
            "mesh_size_inner": mesh_size_inner,
            "mesh_size_outer": mesh_size_outer,
            "mesh_size_transition": mesh_size_transition,
        }
    )
    mesh = mesher.init_mesh()
    boundary_roles = getattr(mesh, "boundary_nodes_by_role", None) or {}
    boundary_nodes_by_role = {
        name: bm.asarray(values, dtype=int).reshape(-1) for name, values in boundary_roles.items()
    }
    design_boundary_node_order = tuple(
        int(value)
        for value in bm.asarray(getattr(mesh, "design_boundary_node_order", ()), dtype=int)
        .reshape(-1)
        .tolist()
    )
    if not design_boundary_node_order:
        raise ValueError("obstacle mesher must attach design boundary metadata")

    design_center = tuple(
        float(value)
        for value in bm.asarray(getattr(mesh, "design_center", center), dtype=float).reshape(-1)[:2]
    )
    design_radius = float(getattr(mesh, "design_radius", radius))
    design_area = float(getattr(mesh, "design_area", 0.0))
    geometry_regularization = ObstacleGeometryRegularization(
        design_node_order=design_boundary_node_order,
        reference_volume=design_area,
        reference_barycenter=bm.asarray(design_center, dtype=float),
        factor_volume=factor_volume,
        factor_barycenter=factor_barycenter,
    )
    propagation_parameters = {
        "boundary_nodes": bm.asarray(getattr(mesh, "boundary_nodes", ()), dtype=int),
        "design_boundary_nodes": bm.asarray(
            getattr(mesh, "design_boundary_ids", design_boundary_node_order), dtype=int
        ),
        "fixed_nodes": bm.asarray(getattr(mesh, "fixed_nodes", ()), dtype=int),
        "shape_bdry_def": (4,),
        "shape_bdry_fix": (1, 2, 3),
        "lambda_lame": 0.0,
        "damping_factor": 0.0,
        "mu_def": mu_def,
        "mu_fix": mu_fix,
        "quality_quantile": 0.0,
        "test_for_intersections": True,
    }
    propagation_parameters.update(_remesh_quality_parameters(remesh_quality_profile))
    boundary_info = {
        "boundary_markers": dict(DEFAULT_BOUNDARY_MARKERS),
        "boundary_nodes_by_role": boundary_nodes_by_role,
        "design_boundary_node_order": design_boundary_node_order,
        "design_center": design_center,
        "design_radius": design_radius,
        "mesh_size": h,
        "mesh_size_profile": mesh_size_profile,
        "propagation_parameters": propagation_parameters,
        "spatial_dim": 2,
    }
    objective_parameters = {
        "q": 4,
        "viscosity": viscosity,
        "factor_volume": factor_volume,
        "factor_barycenter": factor_barycenter,
        "use_initial_volume": True,
        "use_initial_barycenter": True,
        "volume_reference": design_area,
        "barycenter_reference": bm.asarray(design_center, dtype=float),
        "geometry_regularization": geometry_regularization,
        "design_boundary_node_order": design_boundary_node_order,
        "design_boundary_node_ids": design_boundary_node_order,
        "mesh_size_profile": mesh_size_profile,
        "mesh_size_inner": mesh_size_inner,
        "mesh_size_outer": mesh_size_outer,
        "mesh_size_transition": mesh_size_transition,
        "volume_term": geometry_regularization.volume_term,
        "barycenter_term": geometry_regularization.barycenter_term,
        "regularization_term": lambda *args, **kwargs: 0.0,
        "shape_derivative_source": lambda *args, **kwargs: None,
        "adjoint_rhs_scale": 2.0,
    }
    pde = ObstacleStokesFluidModel(box=box, viscosity=viscosity)
    state_solver = StationaryIncompressibleStokesLFEMModel(
        pde=pde,
        mesh=mesh,
        options={**DEFAULT_STATE_OPTIONS, "linear_solver": DEFAULT_LINEAR_SOLVER},
    )
    options = dict(DEFAULT_OPTIONS)
    options.update(
        {
            "initial_step_size": initial_step_size,
            "max_iterations": max_iterations,
            "algorithm": algorithm,
            "rtol": rtol,
        }
    )
    runner = ShapeOptimizationRunner(
        mesh=mesh,
        boundary_info=boundary_info,
        pde=pde,
        state_solver=state_solver,
        options=options,
        objective_parameters=objective_parameters,
        mesher=mesher,
    )
    if runner.cache is not None:
        runner.cache.propagation_parameters["remesh_handler"] = runner.build_remesh_handler()
        runner.cache.propagation_parameters["remesh"] = runner.cache.propagation_parameters["remesh_handler"]

    output_dir = Path.cwd() / f"_tmp_obstacle_stokes_frames_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = runner.run(
        export_vtu=True,
        vtu_output_dir=output_dir,
        vtu_prefix="frame",
        export_initial_state=True,
    )
    print("Obstacle Stokes benchmark configuration summary")
    print(
        {
            "mesh_size": boundary_info["mesh_size"],
            "shape_bdry_def": runner.cache.propagation_parameters.get("shape_bdry_def") if runner.cache is not None else None,
            "shape_bdry_fix": runner.cache.propagation_parameters.get("shape_bdry_fix") if runner.cache is not None else None,
            "algorithm": options.get("algorithm"),
            "rtol": options.get("rtol"),
            "output_dir": str(output_dir),
        }
    )
    print("Obstacle Stokes benchmark run summary")
    print(
        {
            "terminated_reason": result.terminated_reason,
            "iterations_run": result.iterations_run,
            "terminated_early": result.terminated_early,
            "final_objective": result.final_objective,
        }
    )
