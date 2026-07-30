"""Run the Re=20 cylinder-flow benchmark with ``NSFVMSimpleModel``."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    CylinderFlowCase,
    NSFVMSimpleModel,
    steady_ns_high_accuracy_simple_profile,
)
from fealpy.fvm.cylinder_flow_postprocess import write_cylinder_outputs
from fealpy.fvm.benchmark_postprocess import re_label


def default_output_dir(
    re: float,
    *,
    mesh_size: float,
    cylinder_mesh_size: float,
    root: str | Path = "output/cylinder_flow",
) -> Path:
    """Return a stable output directory for one cylinder-flow SIMPLE run."""
    mesh_label = (
        f"tri_h{float(mesh_size):g}_hc{float(cylinder_mesh_size):g}"
        .replace(".", "p")
        .replace("-", "m")
    )
    return Path(root) / "simple" / re_label(re) / mesh_label


def build_case(args) -> CylinderFlowCase:
    """Build the cylinder-flow case from parsed CLI arguments."""
    return CylinderFlowCase(
        re=args.re,
        rho=args.rho,
        mu=args.mu,
        mean_velocity=args.mean_velocity,
        mesh_size=args.mesh_size,
        cylinder_mesh_size=args.cylinder_mesh_size,
        wake_mesh_size=args.wake_mesh_size,
        cylinder_refine_radius=args.cylinder_refine_radius,
        wake_length=args.wake_length,
        wake_half_width=args.wake_half_width,
        outlet_velocity_policy=args.outlet_velocity_policy,
    )


def build_simple_options(case: CylinderFlowCase, args) -> dict:
    """Return ``NSFVMSimpleModel`` options for the cylinder benchmark."""
    base = steady_ns_high_accuracy_simple_profile()
    shared_interpolation = args.face_interpolation_method
    profile = replace(
        base,
        discretization=replace(
            base.discretization,
            pressure_gradient_method=args.pressure_gradient_method,
            velocity_gradient_method=args.velocity_gradient_method,
            momentum_face_interpolation=shared_interpolation,
            pressure_response_interpolation=shared_interpolation,
            rhie_chow_velocity_interpolation=shared_interpolation,
            face_flux_correction_scheme="none",
            face_flux_quadrature_order=3,
        ),
        iteration=replace(
            base.iteration,
            max_iterations=args.max_iter,
            pressure_relaxation=args.relax,
            momentum_equation_relaxation=0.7,
            momentum_relative_tolerance=(
                args.tol
                if args.tol_momentum is None
                else args.tol_momentum
            ),
            mass_relative_tolerance=(
                args.tol if args.tol_mass is None else args.tol_mass
            ),
        ),
    )
    options = {
        "pde": case,
        "mesh_type": "improved_tri",
        "profile": profile,
        "pbar_log": args.pbar_log,
        "log_level": args.log_level,
    }
    if args.engineering_boundary_conditions:
        options["boundary_conditions"] = case.engineering_boundary_conditions
    if args.rho is not None:
        options["rho"] = args.rho
    if args.mu is not None:
        options["mu"] = args.mu
    return options


def run_simple_cylinder(args):
    """Run SIMPLE and write standard cylinder-flow benchmark outputs."""
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    case = build_case(args)
    model = NSFVMSimpleModel(build_simple_options(case, args))
    solve_result = model.solve()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            case.re,
            mesh_size=args.mesh_size,
            cylinder_mesh_size=args.cylinder_mesh_size,
        )
    )
    outputs = write_cylinder_outputs(
        model,
        case,
        solve_result,
        output_dir,
        velocity_gradient=(
            model.solver.spatial_face_velocity.boundary.gradient
        ),
        residuals=solve_result.residual_history,
        run_summary={
            "solver": "NSFVMSimpleModel",
            "re": case.re,
            "rho": case.rho,
            "mu": case.mu,
            "nu": case.nu,
            "mean_velocity": case.mean_velocity,
            "mesh_type": "improved_tri",
            "mesh_size": args.mesh_size,
            "cylinder_mesh_size": args.cylinder_mesh_size,
            "wake_mesh_size": args.wake_mesh_size,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "tol_momentum": args.tol_momentum,
            "tol_mass": args.tol_mass,
            "relax": args.relax,
            "engineering_boundary_conditions": args.engineering_boundary_conditions,
            "pressure_gradient_method": args.pressure_gradient_method,
            "velocity_gradient_method": args.velocity_gradient_method,
            "face_interpolation_method": args.face_interpolation_method,
            "force_viscous_method": args.force_viscous_method,
        },
        viscous_method=args.force_viscous_method,
        fields=tuple(args.output_fields),
    )
    return model, outputs


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cylinder-flow benchmark with NSFVMSimpleModel"
    )
    parser.add_argument("--re", default=20.0, type=float)
    parser.add_argument("--rho", default=1.0, type=float)
    parser.add_argument("--mu", default=None, type=float)
    parser.add_argument("--mean_velocity", default=0.2, type=float)
    parser.add_argument("--mesh_size", default=0.04, type=float)
    parser.add_argument("--cylinder_mesh_size", default=0.006, type=float)
    parser.add_argument("--wake_mesh_size", default=0.02, type=float)
    parser.add_argument("--cylinder_refine_radius", default=None, type=float)
    parser.add_argument("--wake_length", default=None, type=float)
    parser.add_argument("--wake_half_width", default=None, type=float)
    parser.add_argument(
        "--outlet_velocity_policy",
        default="profile",
        choices=("profile", "zero"),
    )
    parser.add_argument(
        "--pressure_gradient_method",
        default="layered_lsq",
        choices=(
            "layered_lsq",
            "face_weighted_lsq",
            "green_gauss",
        ),
    )
    parser.add_argument(
        "--velocity_gradient_method",
        default="layered_lsq",
        choices=(
            "layered_lsq",
            "face_weighted_lsq",
            "green_gauss",
        ),
    )
    parser.add_argument(
        "--face_interpolation_method",
        default="average",
        choices=("average", "linear"),
    )
    parser.add_argument(
        "--force_viscous_method",
        default="wall_sn_grad",
        choices=("wall_sn_grad", "cell_gradient", "none"),
    )
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--tol", default=1.0e-6, type=float)
    parser.add_argument("--tol_mass", default=None, type=float)
    parser.add_argument("--tol_momentum", default=None, type=float)
    parser.add_argument("--relax", default=0.03, type=float)
    parser.add_argument(
        "--engineering_boundary_conditions",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log_level", default="WARNING", type=str)
    parser.add_argument("--pbar_log", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["velocity", "u", "v", "pressure", "speed"],
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    model, outputs = run_simple_cylinder(args)
    print(model)
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
