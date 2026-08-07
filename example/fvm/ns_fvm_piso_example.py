import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    CollocatedPressureSystemControls,
    NSFVMPISOModel,
    PressureClosureKind,
)


def main():
    parser = argparse.ArgumentParser(
        description="PISO-based FVM Navier-Stokes solver with Rhie-Chow interpolation")

    parser.add_argument("--pde", default=3, type=int,
                        help="Navier-Stokes PDE example ID")

    parser.add_argument("--nx", default=20, type=int,
                        help="Grid divisions in x")

    parser.add_argument("--ny", default=20, type=int,
                        help="Grid divisions in y")

    parser.add_argument("--nz", default=None, type=int,
                        help="Grid divisions in z for 3D examples")

    parser.add_argument("--mesh_type", default="uniform_quad", type=str,
                        help="Mesh type, e.g. uniform_quad or uniform_tri")

    parser.add_argument("--time_steps", default=40, type=int,
                        help="Number of time steps")

    parser.add_argument("--n_correctors", default=2, type=int,
                        help="Number of PISO pressure correctors per time step")

    parser.add_argument("--momentum_nonorthogonal_max_iterations", default=None, type=int,
                        help="Max explicit non-orthogonal corrections for momentum diffusion")

    parser.add_argument("--pressure_nonorthogonal_max_iterations", default=None, type=int,
                        help="Max explicit non-orthogonal corrections for pressure correction")

    parser.add_argument("--duration", nargs=2, default=(0.0, 1.0), type=float,
                        help="Start and end time")

    parser.add_argument("--backend", default="numpy", type=str,
                        help="Backend: numpy, torch, tensorflow, or jax.")

    parser.add_argument("--pure_neumann_closure", default="nullspace", type=str,
                        choices=("gauge", "nullspace"),
                        help="Pressure uniqueness treatment for pure-Neumann pressure systems.")

    parser.add_argument("--pbar_log", default=True, type=bool,
                        help="Whether to show progress bar, default is True")

    parser.add_argument("--log_level", default="INFO", type=str,
                        help="Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL")

    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    bm.set_backend(args.backend)
    options = {
        "pde": args.pde,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "mesh_type": args.mesh_type,
        "time_steps": args.time_steps,
        "n_correctors": args.n_correctors,
        "momentum_nonorthogonal_max_iterations": (
            args.momentum_nonorthogonal_max_iterations
        ),
        "pressure_nonorthogonal_max_iterations": (
            args.pressure_nonorthogonal_max_iterations
        ),
        "duration": tuple(args.duration),
        "pressure_system_controls": CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind(
                args.pure_neumann_closure
            ),
        ),
        "pbar_log": args.pbar_log,
        "log_level": args.log_level,
    }

    model = NSFVMPISOModel(options)
    print(model)

    result = model.solve()
    errors = model.compute_error(result)
    velocity_names = ("u", "v", "w")
    for name, error in zip(velocity_names, errors[:-1]):
        print(f"L2 error ({name}) = {error}")
    print(f"L2 error (p) = {errors[-1]}")
    if args.plot:
        model.plot(result)


if __name__ == "__main__":
    main()
