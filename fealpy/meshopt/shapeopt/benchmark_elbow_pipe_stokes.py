"""Direct elbow-pipe-in-NS benchmark script."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.mesher import PiecewiseLinearTransitionPipeMesher2D
from fealpy.model.navier_stokes.exp0008 import ElbowPipeStokesFluidModel

from fealpy.meshopt.shapeopt.benchmark_common import _remesh_quality_parameters
from fealpy.meshopt.shapeopt.benchmark_runner import ShapeOptimizationRunner
from fealpy.meshopt.shapeopt.geometry_regularization import (
    ObstacleGeometryRegularization,
    _polygon_area_and_centroid,
)

DEFAULT_PIPE_PARAMETERS = {
    "channel_width": 0.75,
    "inlet_length": 2.4,
    "bend_length": 3.0,
    "rise_height": 2.55,
    "outlet_length": 4.9,
    "corner_chamfer_ratio": 0.45,
    "design_margin_inlet_ratio": 1.0 - 1.0 / 2.4,
    "design_margin_outlet_ratio": 0.5,
    "mesh_size_global": 0.10,
    "line_samples_per_unit": 8.0,
    "arc_samples": 8,
    "mesh_size_profile": "graded",
    "mesh_size_inner": 0.04,
    "mesh_size_outer": 0.1,
    "mesh_size_transition": 0.5,
}
DEFAULT_INLET_MAX_VELOCITY = 1.5
DEFAULT_REYNOLDS_NUMBER = 400.0
DEFAULT_DENSITY = 1.0
DEFAULT_VISCOSITY = None
DEFAULT_ALGORITHM = "steepest_descent"  # "lbfgs" or "gradient_descent"
DEFAULT_RTOL = 5.0e-4
DEFAULT_INITIAL_STEP_SIZE = 1.0
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_REMESH_QUALITY_PROFILE = "cashocs"
DEFAULT_LINEAR_SOLVER = "mumps"
DEFAULT_FACTOR_VOLUME = 1.0
DEFAULT_BOUNDARY_MARKERS = {
    "inlet": (2,),
    "outlet": (3,),
    "wall": (4,),
    "fixed": (2, 3),
    "design": (4,),
}
DEFAULT_OPTIONS = {
    "q": 4,
    "algorithm": DEFAULT_ALGORITHM,
    "rtol": DEFAULT_RTOL,
    "initial_step_size": DEFAULT_INITIAL_STEP_SIZE,
    "area_preserving_projection": False,
    "armijo_use_finite_difference_directional_derivative": False,
    "armijo_use_finite_difference_directional_derivative_on_reject": True,
    "armijo_fd_step_size": 1.0e-6,
    "line_search_reduction": 0.5,
    "epsilon_armijo": 1.0e-4,
    "minimum_step_size": 1.0e-8,
    "max_line_search_iterations": 20,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "apply_bc": "dirichlet_dof",
    "linear_solver": DEFAULT_LINEAR_SOLVER,
    "save_results": False,
    "save_txt": False,
    "state_system_is_linear": False,
}
DEFAULT_STATE_OPTIONS = {
    "run": "one_step",
    "solve": DEFAULT_LINEAR_SOLVER,
    "method": "Newton",
    "apply_bc": "dirichlet_dof",
}

if __name__ == "__main__":  # pragma: no cover
    pipe_parameters = dict(DEFAULT_PIPE_PARAMETERS)
    channel_width = float(pipe_parameters["channel_width"])
    inlet_length = float(pipe_parameters["inlet_length"])
    bend_length = float(pipe_parameters["bend_length"])
    outlet_length = float(pipe_parameters["outlet_length"])
    mesh_size_profile = str(pipe_parameters["mesh_size_profile"])
    mesh_size_inner = float(pipe_parameters["mesh_size_inner"])
    mesh_size_outer = float(pipe_parameters["mesh_size_outer"])
    mesh_size_transition = float(pipe_parameters["mesh_size_transition"])
    inlet_max_velocity = float(DEFAULT_INLET_MAX_VELOCITY)
    density = float(DEFAULT_DENSITY)
    factor_volume = float(DEFAULT_FACTOR_VOLUME)
    mean_inlet_velocity = (2.0 / 3.0) * inlet_max_velocity
    reference_length = channel_width
    effective_viscosity = density * mean_inlet_velocity * reference_length / float(DEFAULT_REYNOLDS_NUMBER)
    algorithm = DEFAULT_ALGORITHM
    rtol = DEFAULT_RTOL
    initial_step_size = DEFAULT_INITIAL_STEP_SIZE
    max_iterations = DEFAULT_MAX_ITERATIONS
    remesh_quality_profile = DEFAULT_REMESH_QUALITY_PROFILE

    mesher = PiecewiseLinearTransitionPipeMesher2D(pipe_parameters)
    mesh = mesher.init_mesh()
    nodes = mesh.entity("node")
    boundary_roles = getattr(mesh, "boundary_nodes_by_role", None) or {}
    boundary_nodes_by_role = {
        name: bm.asarray(values, dtype=int).reshape(-1) for name, values in boundary_roles.items()
    }
    boundary_cycle = bm.asarray(getattr(mesh, "boundary_nodes", ()), dtype=int).reshape(-1)
    design_boundary_node_order = tuple(
        int(node) for node in bm.asarray(getattr(mesh, "design_boundary_node_order", boundary_cycle), dtype=int).reshape(-1).tolist()
    )
    if not design_boundary_node_order:
        raise ValueError("elbow pipe mesher must attach design_boundary_node_order metadata")
    design_nodes = bm.asarray(getattr(mesh, "design_nodes", design_boundary_node_order), dtype=int).reshape(-1)
    fixed_nodes = bm.asarray(getattr(mesh, "fixed_nodes", ()), dtype=int).reshape(-1)
    inlet_nodes = bm.asarray(getattr(mesh, "inlet_nodes", ()), dtype=int).reshape(-1)
    outlet_nodes = bm.asarray(getattr(mesh, "outlet_nodes", ()), dtype=int).reshape(-1)
    wall_nodes = bm.asarray(getattr(mesh, "wall_nodes", ()), dtype=int).reshape(-1)
    fixed_x_left = float(getattr(mesh, "fixed_x_left", inlet_length * (1.0 - float(pipe_parameters["design_margin_inlet_ratio"]))))
    fixed_x_right = float(getattr(mesh, "fixed_x_right", inlet_length + bend_length + outlet_length * float(pipe_parameters["design_margin_outlet_ratio"])))
    inlet_x = float(bm.mean(nodes[inlet_nodes, 0])) if inlet_nodes.size else float(bm.min(nodes[:, 0]))
    inlet_ymin = float(bm.min(nodes[inlet_nodes, 1])) if inlet_nodes.size else float(bm.min(nodes[:, 1]))
    inlet_ymax = float(bm.max(nodes[inlet_nodes, 1])) if inlet_nodes.size else float(bm.max(nodes[:, 1]))
    outlet_x = float(bm.mean(nodes[outlet_nodes, 0])) if outlet_nodes.size else float(bm.max(nodes[:, 0]))
    design_coords = nodes[bm.asarray(design_boundary_node_order, dtype=int)]
    design_center = tuple(float(v) for v in bm.mean(design_coords, axis=0).reshape(-1)[:2])
    design_radius = float(bm.mean(bm.linalg.norm(design_coords - bm.asarray(design_center, dtype=float), axis=1)))
    design_area, design_barycenter = _polygon_area_and_centroid(
        design_coords,
        fallback_center=design_center,
    )
    geometry_regularization = ObstacleGeometryRegularization(
        design_node_order=design_boundary_node_order,
        reference_volume=float(design_area),
        reference_barycenter=bm.asarray(design_barycenter, dtype=float),
        factor_volume=factor_volume,
        factor_barycenter=0.0,
    )
    boundary_markers = dict(DEFAULT_BOUNDARY_MARKERS)
    geometry_contract = None
    propagation_parameters = {
        "boundary_nodes": boundary_cycle,
        "design_boundary_nodes": design_nodes,
        "fixed_nodes": fixed_nodes,
        "shape_bdry_def": (4,),
        "shape_bdry_fix": (2, 3),
        "fixed_x_left": fixed_x_left,
        "fixed_x_right": fixed_x_right,
        "lambda_lame": 0.0,
        "damping_factor": 0.0,
        "mu_def": 5.0e2,
        "mu_fix": 1.0,
        "quality_quantile": 0.0,
        "test_for_intersections": True,
    }
    propagation_parameters.update(_remesh_quality_parameters(remesh_quality_profile))
    boundary_info = {
        "boundary_markers": boundary_markers,
        "boundary_nodes_by_role": boundary_nodes_by_role,
        "design_boundary_node_order": design_boundary_node_order,
        "design_center": design_center,
        "design_radius": design_radius,
        "mesh_size": float(pipe_parameters["mesh_size_global"]),
        "mesh_size_profile": mesh_size_profile,
        "fixed_x_left": fixed_x_left,
        "fixed_x_right": fixed_x_right,
        "spatial_dim": 2,
        "propagation_parameters": propagation_parameters,
    }
    objective_parameters = {
        "q": 4,
        "viscosity": effective_viscosity,
        "factor_volume": factor_volume,
        "use_initial_volume": True,
        "volume_reference": float(design_area),
        "geometry_regularization": geometry_regularization,
        "reynolds_number": float(DEFAULT_REYNOLDS_NUMBER),
        "inlet_max_velocity": inlet_max_velocity,
        "density": density,
        "reference_length": reference_length,
        "design_boundary_node_order": design_boundary_node_order,
        "design_boundary_node_ids": design_boundary_node_order,
        "mesh_size_profile": mesh_size_profile,
        "mesh_size_inner": mesh_size_inner,
        "mesh_size_outer": mesh_size_outer,
        "mesh_size_transition": mesh_size_transition,
        "fixed_x_left": fixed_x_left,
        "fixed_x_right": fixed_x_right,
        "adjoint_rhs_scale": 2.0,
        "volume_term": geometry_regularization.volume_term,
        "shape_derivative_source": lambda *args, **kwargs: None,
        "regularization_term": lambda *args, **kwargs: 0.0,
    }
    
    pde = ElbowPipeStokesFluidModel(
        inlet_x=inlet_x,
        inlet_ymin=inlet_ymin,
        inlet_ymax=inlet_ymax,
        outlet_x=outlet_x,
        rho=density,
        viscosity=effective_viscosity,
        inlet_max_velocity=inlet_max_velocity,
        pressure_neumann=True,
    )
    pde.mesh = mesh
    state_solver = StationaryIncompressibleNSLFEMModel(
        pde=pde,
        mesh=mesh,
        options={**DEFAULT_STATE_OPTIONS, "solve": DEFAULT_LINEAR_SOLVER},
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

    output_dir = Path.cwd() / f"_tmp_elbow_stokes_frames_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = runner.run(export_vtu=True, vtu_output_dir=output_dir, vtu_prefix="frame")
    print("Elbow pipe benchmark configuration summary")
    print(
        {
            "reynolds_number": DEFAULT_REYNOLDS_NUMBER,
            "effective_viscosity": effective_viscosity,
            "factor_volume": factor_volume,
            "area_preserving_projection": options["area_preserving_projection"],
            "mesh_size": boundary_info["mesh_size"],
            "algorithm": options["algorithm"],
            "rtol": options["rtol"],
            "output_dir": str(output_dir),
        }
    )
    print("Elbow pipe benchmark run summary")
    print(
        {
            "terminated_reason": result.terminated_reason,
            "iterations_run": result.iterations_run,
            "terminated_early": result.terminated_early,
            "final_objective": result.final_objective,
        }
    )
