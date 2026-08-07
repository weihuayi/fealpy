"""Run the lid-driven cavity benchmark with ``NSFVMPISOModel``."""

from __future__ import annotations

import argparse
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.fvm import LidDrivenCavityCase, NSFVMPISOModel
from fealpy.fvm.lid_driven_cavity_postprocess import (
    CavityOutputConfig,
    CavitySnapshotWriter,
    default_output_dir,
    write_benchmark_outputs,
)


def build_piso_options(
    case: LidDrivenCavityCase,
    *,
    nx: int,
    ny: int,
    mesh_type: str,
    time_steps: int,
    duration: tuple[float, float],
    n_correctors: int,
    backend: str,
    log_level: str,
    rho: float | None = None,
    mu: float | None = None,
    pbar_log: bool = False,
    device: str = "cpu",
) -> dict:
    options = {
        "pde": case,
        "nx": int(nx),
        "ny": int(ny),
        "mesh_type": mesh_type,
        "duration": tuple(duration),
        "time_steps": int(time_steps),
        "n_correctors": int(n_correctors),
        "pbar_log": pbar_log,
        "log_level": log_level,
    }
    if rho is not None:
        options["rho"] = float(rho)
    if mu is not None:
        options["mu"] = float(mu)
    return options


def run_piso_cavity(args):
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    duration = (float(args.duration[0]), float(args.duration[1]))
    case = LidDrivenCavityCase(
        re=args.re,
        lid_velocity=args.lid_velocity,
        rho=args.rho,
        mu=args.mu,
    )
    options = build_piso_options(
        case,
        nx=args.nx,
        ny=args.ny,
        mesh_type=args.mesh_type,
        time_steps=args.time_steps,
        duration=duration,
        n_correctors=args.n_correctors,
        backend=args.backend,
        device=args.device,
        log_level=args.log_level,
        rho=args.rho,
        mu=args.mu,
        pbar_log=args.pbar_log,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir("piso", args.re, args.mesh_type, args.nx, args.ny)
    )
    output_config = CavityOutputConfig(
        output_dir=output_dir,
        write_vtk=args.write_time_snapshots,
        write_final=args.write_final_snapshot,
        write_interval_steps=args.write_interval_steps,
        write_interval_time=args.write_interval_time,
        fields=tuple(args.output_fields),
    )
    model = NSFVMPISOModel(options)
    snapshot_writer = CavitySnapshotWriter(
        output_config,
        geometry=model.fvm_geometry,
        velocity_gradient=(
            model.solver.spatial_face_velocity.boundary.gradient
        ),
        domain=tuple(case.domain()),
        total_steps=args.time_steps,
    )

    result = model.solve(snapshot_callback=snapshot_writer)
    snapshot_writer.write_time_history()

    outputs = write_benchmark_outputs(
        model,
        output_dir,
        velocity=result.velocity,
        pressure=result.pressure,
        domain=tuple(case.domain()),
        run_summary={
            "solver": "NSFVMPISOModel",
            "re": case.re,
            "rho": case.rho,
            "mu": case.mu,
            "nu": case.nu,
            "nx": args.nx,
            "ny": args.ny,
            "mesh_type": args.mesh_type,
            "duration": duration,
            "time_steps": args.time_steps,
            "n_correctors": args.n_correctors,
            "write_interval_steps": args.write_interval_steps,
            "write_interval_time": args.write_interval_time,
            "write_final_snapshot": args.write_final_snapshot,
            "coefficient_note": "rho and mu are applied only in the momentum equation",
        },
        fields=tuple(args.output_fields),
    )
    return model, outputs


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lid-driven cavity benchmark with NSFVMPISOModel"
    )
    parser.add_argument("--re", default=1.0, type=float)
    parser.add_argument("--lid_velocity", default=1.0, type=float)
    parser.add_argument("--rho", default=1.0, type=float)
    parser.add_argument("--mu", default=None, type=float)
    parser.add_argument("--nx", default=32, type=int)
    parser.add_argument("--ny", default=32, type=int)
    parser.add_argument("--mesh_type", default="uniform_quad", type=str)
    parser.add_argument("--time_steps", default=200, type=int)
    parser.add_argument("--duration", nargs=2, default=(0.0, 5.0), type=float)
    parser.add_argument("--n_correctors", default=4, type=int)
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log_level", default="WARNING", type=str)
    parser.add_argument("--pbar_log", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--write_interval_steps", default=None, type=int)
    parser.add_argument("--write_interval_time", default=None, type=float)
    parser.add_argument(
        "--write_time_snapshots",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--write_final_snapshot",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["velocity", "u", "v", "pressure", "speed"],
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    model, outputs = run_piso_cavity(args)
    print(model)
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Primary vortex estimate: {outputs['vortex']}")


if __name__ == "__main__":
    main()
