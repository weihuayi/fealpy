import argparse
from dataclasses import replace

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    StokesFVMSimpleModel,
    steady_ns_high_accuracy_simple_profile,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solve a Stokes manufactured solution with SIMPLE."
    )
    parser.add_argument(
        "--pde",
        default=1,
        type=int,
        help="Stokes PDE example ID.",
    )
    parser.add_argument(
        "--mesh-type",
        default=None,
        help="Override the mesh type supplied by the PDE model.",
    )
    parser.add_argument(
        "--mesh-refine",
        default=3,
        type=int,
        help="Uniform refinement levels applied to the PDE mesh.",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="FEALPy backend, such as numpy or pytorch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used by the selected backend.",
    )
    parser.add_argument("--max-iter", default=3000, type=int)
    parser.add_argument("--tol", default=1.0e-6, type=float)
    parser.add_argument(
        "--relax",
        default=0.3,
        type=float,
        help="Pressure-correction relaxation factor.",
    )
    parser.add_argument(
        "--momentum-relaxation",
        default=0.9,
        type=float,
        help="Momentum-equation under-relaxation factor.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration SIMPLE diagnostics.",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    if args.device != "cpu" and args.backend != "pytorch":
        raise ValueError("GPU execution is currently supported through pytorch backend.")
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    base_profile = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base_profile,
        iteration=replace(
            base_profile.iteration,
            max_iterations=args.max_iter,
            pressure_relaxation=args.relax,
            momentum_equation_relaxation=args.momentum_relaxation,
            momentum_relative_tolerance=args.tol,
            mass_relative_tolerance=args.tol,
        ),
    )
    model_options = {
        "pde": args.pde,
        "mesh_refine": args.mesh_refine,
        "log_level": "WARNING" if args.quiet else "INFO",
        "profile": profile,
    }
    if args.mesh_type is not None:
        model_options["mesh_type"] = args.mesh_type

    model = StokesFVMSimpleModel(model_options)
    print(model)

    result = model.solve()
    status = "converged" if result.converged else "not converged"
    print(
        f"SIMPLE {status} after {result.outer_iterations} iterations "
        f"({result.termination_reason})."
    )

    errors = model.compute_error(result)
    for name, error in zip(("u", "v", "w"), errors[:-1]):
        print(f"L2 error ({name}) = {error}")
    print(f"L2 error (p) = {errors[-1]}")

    if args.plot:
        model.plot(result)
        model.plot_residual(result)


if __name__ == "__main__":
    main()
