"""Cylinder-flow PISO benchmark driver for FEALPy FVM.

This example has two modes:

* default mode runs one transient cylinder-flow PISO solve;
* ``--grid_convergence`` runs several mesh levels and estimates successive
  orders from scalar benchmark quantities.

The numerical model, mesh generation, boundary conditions, solver, and
post-processing helpers live in ``fealpy.fvm``.  This file only organizes the
benchmark run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    CylinderFlowCase,
    NSFVMPISOModel,
    PisoCorrectorDiagnostic,
    PisoSnapshot,
)
from fealpy.fvm.cylinder_flow_postprocess import (
    CylinderSolutionFields,
    cylinder_force_coefficients,
    pressure_drop,
    write_cylinder_outputs,
)
from fealpy.fvm.benchmark_postprocess import (
    re_label,
    scalarize_rows,
    write_dict_csv,
    write_solution_vtk,
)


DEFAULT_GRID_LEVELS = (
    "0.12:0.03:0.06",
    "0.06:0.015:0.03",
    "0.03:0.0075:0.015",
)

DEFAULT_GRID_METRICS = (
    "force_drag_coefficient",
    "force_lift_coefficient",
    "pressure_drop_delta_p",
    "force_pressure_force_x",
    "force_viscous_force_x",
)


def default_output_dir(
    re: float,
    *,
    mesh_size: float,
    cylinder_mesh_size: float,
    root: str | Path = "output/cylinder_flow",
) -> Path:
    mesh_label = (
        f"tri_h{float(mesh_size):g}_hc{float(cylinder_mesh_size):g}"
        .replace(".", "p")
        .replace("-", "m")
    )
    return Path(root) / "piso" / re_label(re) / mesh_label


def build_case(args) -> CylinderFlowCase:
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


def build_piso_options(case: CylinderFlowCase, args) -> dict:
    options = {
        "pde": case,
        "mesh_type": "improved_tri",
        "duration": tuple(float(value) for value in args.duration),
        "time_steps": int(args.time_steps),
        "n_correctors": int(args.n_correctors),
        "pbar_log": args.pbar_log,
        "log_level": args.log_level,
        "pressure_gradient_method": args.pressure_gradient_method,
        "velocity_gradient_method": args.velocity_gradient_method,
        "momentum_face_interpolation": (
            args.momentum_face_interpolation
        ),
        "pressure_response_interpolation": (
            args.pressure_response_interpolation
        ),
        "rhie_chow_velocity_interpolation": (
            args.rhie_chow_velocity_interpolation
        ),
        "momentum_nonorthogonal_max_iterations": (
            args.momentum_nonorthogonal_max_iterations
        ),
        "pressure_nonorthogonal_max_iterations": (
            args.pressure_nonorthogonal_max_iterations
        ),
        "use_transient_flux_correction": args.use_transient_flux_correction,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_start_step": args.snapshot_start_step,
    }
    if args.engineering_boundary_conditions:
        options["boundary_conditions"] = case.engineering_boundary_conditions
    if args.rho is not None:
        options["rho"] = args.rho
    if args.mu is not None:
        options["mu"] = args.mu
    return options


class CylinderPISOHistory:
    """Collect transient diagnostics and optional VTU snapshots."""

    def __init__(
        self,
        case: CylinderFlowCase,
        *,
        model: NSFVMPISOModel,
        output_dir: Path,
        fields: tuple[str, ...],
        viscous_method: str,
        vtk_interval: int = 0,
        vtk_start_step: int = 1,
    ):
        if vtk_interval < 0:
            raise ValueError("vtk_interval must be non-negative.")
        if vtk_start_step < 1:
            raise ValueError("vtk_start_step must be positive.")
        self.case = case
        self.model = model
        self.geometry = model.fvm_geometry
        self.velocity_gradient = (
            model.solver.spatial_face_velocity.boundary.gradient
        )
        self.output_dir = Path(output_dir)
        self.fields = fields
        self.viscous_method = viscous_method
        self.vtk_interval = int(vtk_interval)
        self.vtk_start_step = int(vtk_start_step)
        self.rows: list[dict] = []
        self.force_rows: list[dict] = []
        self.corrector_rows: list[PisoCorrectorDiagnostic] = []
        self._previous_velocity = None

    def __call__(
        self,
        snapshot: PisoSnapshot,
    ) -> None:
        step = snapshot.step
        time = snapshot.time
        cell_velocity = snapshot.velocity
        pressure = snapshot.pressure
        flux = snapshot.face_flux
        speed = bm.sqrt(cell_velocity[:, 0] ** 2 + cell_velocity[:, 1] ** 2)
        row = {
            "step": int(step),
            "time": float(time),
            "mass": self._mass_residual(flux),
            "velocity_update": self._velocity_update(cell_velocity),
            "speed_max": self._scalar(bm.max(speed)),
            "speed_mean": self._scalar(bm.mean(speed)),
        }
        row.update(
            self._outlet_flux_diagnostics(
                self.model,
                self.case,
                flux,
            )
        )
        self.rows.append(row)

        force = cylinder_force_coefficients(
            self.model.mesh,
            self.case,
            velocity=cell_velocity,
            pressure=pressure,
            velocity_gradient=self.velocity_gradient,
            viscous_method=self.viscous_method,
            geometry=self.geometry,
        )
        probes = pressure_drop(
            self.model.mesh.entity_barycenter("cell"),
            pressure,
        )
        self.force_rows.append(
            {
                "step": int(step),
                "time": float(time),
                **force,
                **{f"pressure_drop_{key}": value for key, value in probes.items()},
            }
        )

        if (
            self.vtk_interval > 0
            and step >= self.vtk_start_step
            and (step - self.vtk_start_step) % self.vtk_interval == 0
        ):
            write_solution_vtk(
                self.model.mesh,
                cell_velocity,
                pressure,
                self.output_dir / "snapshots" / f"solution_{int(step):06d}.vtu",
                fields=self.fields,
                velocity_gradient=self.velocity_gradient,
            )
        self._previous_velocity = bm.array(cell_velocity)

    def record_corrector(
        self,
        row: PisoCorrectorDiagnostic,
    ) -> None:
        self.corrector_rows.append(row)

    def _velocity_update(self, cell_velocity):
        if self._previous_velocity is None:
            return 0.0
        return self._scalar(bm.max(bm.abs(cell_velocity - self._previous_velocity)))

    def _mass_residual(self, flux):
        if flux is None:
            return None
        return self._scalar(
            bm.max(
                bm.abs(
                    self.geometry.scatter_face_flux_to_cells(flux)
                )
            )
        )

    @classmethod
    def _outlet_flux_diagnostics(cls, model, case, flux):
        if flux is None:
            return {
                "outlet_flux_total": None,
                "outlet_flux_min": None,
                "outlet_backflow_flux": None,
                "outlet_backflow_face_count": None,
            }

        boundary_faces = model.mesh.boundary_face_index()
        face_centers = model.mesh.entity_barycenter("face")[boundary_faces]
        outlet_faces = boundary_faces[case.is_outlet_boundary(face_centers)]
        if outlet_faces.shape[0] == 0:
            return {
                "outlet_flux_total": 0.0,
                "outlet_flux_min": 0.0,
                "outlet_backflow_flux": 0.0,
                "outlet_backflow_face_count": 0,
            }

        outlet_flux = flux[outlet_faces]
        backflow = bm.maximum(-outlet_flux, 0.0)
        return {
            "outlet_flux_total": cls._scalar(bm.sum(outlet_flux)),
            "outlet_flux_min": cls._scalar(bm.min(outlet_flux)),
            "outlet_backflow_flux": cls._scalar(bm.sum(backflow)),
            "outlet_backflow_face_count": int(cls._scalar(bm.sum(outlet_flux < 0.0))),
        }

    @staticmethod
    def _scalar(value):
        array = bm.to_numpy(value)
        return float(array.item() if array.shape == () else array)


def run_piso_cylinder(args):
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    case = build_case(args)
    model = NSFVMPISOModel(build_piso_options(case, args))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            case.re,
            mesh_size=args.mesh_size,
            cylinder_mesh_size=args.cylinder_mesh_size,
        )
    )
    output_fields = tuple(args.output_fields)
    history = CylinderPISOHistory(
        case,
        model=model,
        output_dir=output_dir,
        fields=output_fields,
        viscous_method=args.force_viscous_method,
        vtk_interval=args.vtk_interval,
        vtk_start_step=args.vtk_start_step,
    )
    result = model.solve(
        snapshot_callback=history,
        corrector_callback=(
            history.record_corrector if args.piso_corrector_diagnostics else None
        ),
    )

    corrector_diagnostics_path = None
    if args.piso_corrector_diagnostics:
        corrector_diagnostics_path = output_dir / "piso_corrector_diagnostics.csv"
        write_dict_csv(
            corrector_diagnostics_path,
            scalarize_rows(
                [asdict(row) for row in history.corrector_rows]
            ),
        )

    outputs = write_cylinder_outputs(
        model,
        case,
        CylinderSolutionFields(
            velocity=result.velocity,
            pressure=result.pressure,
        ),
        output_dir,
        velocity_gradient=(
            model.solver.spatial_face_velocity.boundary.gradient
        ),
        residuals=history.rows,
        force_history=history.force_rows,
        strouhal_start_time=args.strouhal_start_time,
        strouhal_min_lift_amplitude=args.strouhal_min_lift_amplitude,
        run_summary={
            "solver": "NSFVMPISOModel",
            "re": case.re,
            "rho": case.rho,
            "mu": case.mu,
            "nu": case.nu,
            "mean_velocity": case.mean_velocity,
            "mesh_type": "improved_tri",
            "mesh_size": args.mesh_size,
            "cylinder_mesh_size": args.cylinder_mesh_size,
            "wake_mesh_size": args.wake_mesh_size,
            "duration": tuple(float(value) for value in args.duration),
            "time_steps": args.time_steps,
            "n_correctors": args.n_correctors,
            "snapshot_interval": args.snapshot_interval,
            "snapshot_start_step": args.snapshot_start_step,
            "vtk_interval": args.vtk_interval,
            "vtk_start_step": args.vtk_start_step,
            "strouhal_start_time": args.strouhal_start_time,
            "strouhal_min_lift_amplitude": args.strouhal_min_lift_amplitude,
            "momentum_nonorthogonal_max_iterations": (
                args.momentum_nonorthogonal_max_iterations
            ),
            "pressure_nonorthogonal_max_iterations": (
                args.pressure_nonorthogonal_max_iterations
            ),
            "engineering_boundary_conditions": args.engineering_boundary_conditions,
            "pressure_gradient_method": args.pressure_gradient_method,
            "velocity_gradient_method": args.velocity_gradient_method,
            "momentum_face_interpolation": (
                args.momentum_face_interpolation
            ),
            "pressure_response_interpolation": (
                args.pressure_response_interpolation
            ),
            "rhie_chow_velocity_interpolation": (
                args.rhie_chow_velocity_interpolation
            ),
            "piso_corrector_diagnostics": args.piso_corrector_diagnostics,
            "force_viscous_method": args.force_viscous_method,
        },
        viscous_method=args.force_viscous_method,
        fields=output_fields,
    )
    if corrector_diagnostics_path is not None:
        outputs["piso_corrector_diagnostics"] = corrector_diagnostics_path
    return model, outputs


def parse_grid_level(level: str) -> dict:
    parts = level.split(":")
    if len(parts) != 3:
        raise ValueError("grid level must be 'mesh_size:cylinder_size:wake_size'.")
    mesh_size, cylinder_mesh_size, wake_mesh_size = (float(part) for part in parts)
    if mesh_size <= 0.0 or cylinder_mesh_size <= 0.0 or wake_mesh_size <= 0.0:
        raise ValueError("all grid level sizes must be positive.")
    return {
        "mesh_size": mesh_size,
        "cylinder_mesh_size": cylinder_mesh_size,
        "wake_mesh_size": wake_mesh_size,
    }


def grid_level_label(index: int, level: dict) -> str:
    def label(name: str, value: float) -> str:
        text = f"{float(value):g}".replace(".", "p").replace("-", "m")
        return f"{name}{text}"

    return "_".join(
        [
            f"level{index:02d}",
            label("h", level["mesh_size"]),
            label("hc", level["cylinder_mesh_size"]),
            label("hw", level["wake_mesh_size"]),
        ]
    )


def grid_convergence_rows(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: float(row["mesh_size"]), reverse=True)

    result = []
    for first, second, third in zip(ok_rows, ok_rows[1:], ok_rows[2:]):
        h1 = float(first["mesh_size"])
        h2 = float(second["mesh_size"])
        h3 = float(third["mesh_size"])
        ratio12 = h1 / h2
        ratio23 = h2 / h3
        ratio_consistent = math.isclose(
            ratio12, ratio23, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
        for metric in metrics:
            q1 = _float_or_none(first.get(metric))
            q2 = _float_or_none(second.get(metric))
            q3 = _float_or_none(third.get(metric))
            if q1 is None or q2 is None or q3 is None:
                continue
            diff12 = abs(q1 - q2)
            diff23 = abs(q2 - q3)
            order = ""
            if ratio_consistent and diff12 > 0.0 and diff23 > 0.0:
                order = math.log(diff12 / diff23) / math.log(ratio12)
            result.append(
                {
                    "metric": metric,
                    "level_coarse": first["level"],
                    "level_mid": second["level"],
                    "level_fine": third["level"],
                    "h_coarse": h1,
                    "h_mid": h2,
                    "h_fine": h3,
                    "ratio_coarse_to_mid": ratio12,
                    "ratio_mid_to_fine": ratio23,
                    "ratio_consistent": ratio_consistent,
                    "successive_difference_coarse_mid": diff12,
                    "successive_difference_mid_fine": diff23,
                    "successive_order": order,
                }
            )
    return result


def run_grid_convergence(args) -> dict:
    output_root = Path(args.output_dir or "output/cylinder_flow_piso_grid_convergence")
    output_root.mkdir(parents=True, exist_ok=True)
    levels = [parse_grid_level(level) for level in args.levels]
    rows = []

    for index, level in enumerate(levels, start=1):
        case_args = argparse.Namespace(**vars(args))
        case_args.grid_convergence = False
        case_args.mesh_size = level["mesh_size"]
        case_args.cylinder_mesh_size = level["cylinder_mesh_size"]
        case_args.wake_mesh_size = level["wake_mesh_size"]
        level_label = grid_level_label(index, level)
        case_args.output_dir = str(output_root / level_label)

        row = {
            "level": index,
            "level_label": level_label,
            "mesh_size": level["mesh_size"],
            "cylinder_mesh_size": level["cylinder_mesh_size"],
            "wake_mesh_size": level["wake_mesh_size"],
            "output_dir": case_args.output_dir,
        }
        try:
            _, outputs = run_piso_cylinder(case_args)
            row["status"] = "ok"
            row.update(outputs.get("summary", {}))
        except Exception as exc:
            if args.stop_on_failure:
                raise
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    summary_rows = scalarize_rows(_normalize_rows(rows))
    convergence = scalarize_rows(
        _normalize_rows(grid_convergence_rows(rows, tuple(args.grid_metrics)))
    )
    summary_csv = output_root / "grid_summary.csv"
    summary_json = output_root / "grid_summary.json"
    convergence_csv = output_root / "grid_convergence.csv"
    convergence_json = output_root / "grid_convergence.json"

    write_dict_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    write_dict_csv(convergence_csv, convergence)
    convergence_json.write_text(json.dumps(convergence, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": output_root,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "convergence_csv": convergence_csv,
        "convergence_json": convergence_json,
        "rows": summary_rows,
        "convergence_rows": convergence,
    }


def _normalize_rows(rows: list[dict]) -> list[dict]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _float_or_none(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cylinder-flow transient PISO benchmark with FEALPy FVM"
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
    parser.add_argument("--time_steps", default=400, type=int)
    parser.add_argument("--duration", nargs=2, default=(0.0, 20.0), type=float)
    parser.add_argument("--n_correctors", default=4, type=int)
    parser.add_argument("--snapshot_interval", default=1, type=int)
    parser.add_argument("--snapshot_start_step", default=1, type=int)
    parser.add_argument("--vtk_interval", default=0, type=int)
    parser.add_argument("--vtk_start_step", default=1, type=int)
    parser.add_argument("--strouhal_start_time", default=None, type=float)
    parser.add_argument("--strouhal_min_lift_amplitude", default=1.0e-3, type=float)
    parser.add_argument(
        "--momentum_nonorthogonal_max_iterations",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--pressure_nonorthogonal_max_iterations",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--pressure_gradient_method",
        default="layered_lsq",
        choices=("layered_lsq", "face_weighted_lsq", "green_gauss"),
    )
    parser.add_argument(
        "--velocity_gradient_method",
        default="layered_lsq",
        choices=("layered_lsq", "face_weighted_lsq", "green_gauss"),
    )
    parser.add_argument(
        "--momentum_face_interpolation",
        default="average",
        choices=("average", "linear"),
    )
    parser.add_argument(
        "--pressure_response_interpolation",
        default="average",
        choices=("average", "linear"),
    )
    parser.add_argument(
        "--rhie_chow_velocity_interpolation",
        default="average",
        choices=("average", "linear"),
    )
    parser.add_argument(
        "--force_viscous_method",
        default="wall_sn_grad",
        choices=("wall_sn_grad", "cell_gradient", "none"),
    )
    parser.add_argument(
        "--engineering_boundary_conditions",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use_transient_flux_correction",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--piso_corrector_diagnostics",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--grid_convergence",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run all mesh levels and estimate successive-difference orders.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_GRID_LEVELS),
        help="Grid levels as mesh_size:cylinder_mesh_size:wake_mesh_size.",
    )
    parser.add_argument(
        "--grid_metrics",
        nargs="+",
        default=list(DEFAULT_GRID_METRICS),
        help="Scalar summary metrics used for grid-convergence orders.",
    )
    parser.add_argument("--stop_on_failure", default=False, action="store_true")
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log_level", default="WARNING", type=str)
    parser.add_argument(
        "--pbar_log",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["velocity", "u", "v", "pressure", "speed"],
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.grid_convergence:
        outputs = run_grid_convergence(args)
        print(f"Grid convergence output directory: {outputs['output_dir']}")
        print(f"Summary: {outputs['summary_csv']}")
        print(f"Convergence: {outputs['convergence_csv']}")
        return

    model, outputs = run_piso_cylinder(args)
    print(model)
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
