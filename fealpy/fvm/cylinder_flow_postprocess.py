"""Post-processing helpers for the 2D cylinder-flow FVM benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .benchmark_postprocess import (
    scalarize_rows,
    write_dict_csv,
    write_solution_vtk,
)
from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class CylinderSolutionFields:
    velocity: TensorLike
    pressure: TensorLike


def _as_numpy(values: TensorLike) -> np.ndarray:
    return np.asarray(bm.to_numpy(values))


def _as_float(value) -> float:
    array = np.asarray(bm.to_numpy(value))
    return float(array.item() if array.shape == () else array)


def nearest_cell_index(points: TensorLike, target: tuple[float, float]) -> int:
    """Return the nearest cell-center index to a physical target point."""
    point_array = _as_numpy(points)
    target_array = np.asarray(target, dtype=float)
    distance = np.linalg.norm(point_array - target_array, axis=1)
    return int(np.argmin(distance))


def pressure_drop(
    points: TensorLike,
    pressure: TensorLike,
    *,
    upstream_point: tuple[float, float] = (0.15, 0.2),
    downstream_point: tuple[float, float] = (0.25, 0.2),
) -> dict[str, float | int]:
    """Return DFG-style pressure drop between two cylinder-side probes."""
    upstream_cell = nearest_cell_index(points, upstream_point)
    downstream_cell = nearest_cell_index(points, downstream_point)
    pressure_array = _as_numpy(pressure)
    upstream_pressure = float(pressure_array[upstream_cell])
    downstream_pressure = float(pressure_array[downstream_cell])
    return {
        "upstream_cell": upstream_cell,
        "downstream_cell": downstream_cell,
        "upstream_pressure": upstream_pressure,
        "downstream_pressure": downstream_pressure,
        "delta_p": upstream_pressure - downstream_pressure,
    }


def _wall_velocity(case, points: TensorLike) -> TensorLike:
    """Return cylinder wall velocity for force post-processing."""
    if hasattr(case, "cylinder_wall_velocity"):
        return case.cylinder_wall_velocity(points)
    return bm.zeros(
        points.shape,
        dtype=points.dtype,
        device=bm.get_device(points),
    )


def _wall_sn_grad_viscous_force(
    geometry: FVMGeometry,
    case,
    cylinder_faces: TensorLike,
    owner: TensorLike,
    sf: TensorLike,
    velocity: TensorLike,
) -> TensorLike:
    """Return wall viscous force using a one-sided normal gradient.

    This matches the force-postprocessing convention used by OpenFOAM wall
    patches more closely than sampling the owner-cell reconstructed gradient.
    """
    face_centers = geometry.face_center[cylinder_faces]
    cell_centers = geometry.cell_center[owner]
    area = bm.sqrt(bm.einsum("ij,ij->i", sf, sf))
    normal = sf / area[:, None]
    normal_distance = bm.abs(
        bm.einsum("ij,ij->i", face_centers - cell_centers, normal)
    )
    if bool(bm.to_numpy(bm.any(normal_distance <= 0.0))):
        raise ValueError("Cylinder wall normal distance must be positive.")

    delta_velocity = _wall_velocity(case, face_centers) - velocity[owner]
    grad = delta_velocity[:, :, None] * normal[:, None, :] / normal_distance[:, None, None]
    strain = grad + bm.swapaxes(grad, 1, 2)
    divergence = grad[:, 0, 0] + grad[:, 1, 1]
    identity = bm.eye(2, dtype=sf.dtype)
    deviatoric_strain = strain - (2.0 / 3.0) * divergence[:, None, None] * identity
    traction = bm.einsum("nij,nj->ni", deviatoric_strain, sf)
    return -float(case.mu) * traction


def _cell_gradient_viscous_force(
    case,
    owner: TensorLike,
    sf: TensorLike,
    velocity: TensorLike,
    velocity_gradient,
) -> TensorLike:
    """Return viscous force from the owner-cell reconstructed gradient."""
    grad = velocity_gradient.cell_gradient(velocity)[owner]
    strain = grad + bm.swapaxes(grad, 1, 2)
    traction = bm.einsum("nij,nj->ni", strain, sf)
    return -float(case.mu) * traction


def cylinder_force_coefficients(
    mesh,
    case,
    *,
    velocity: TensorLike,
    pressure: TensorLike,
    velocity_gradient=None,
    viscous_method: str = "wall_sn_grad",
    geometry: FVMGeometry | None = None,
) -> dict[str, float | int]:
    """Integrate pressure and viscous force over the cylinder boundary.

    The face area vector is the owner-cell outward vector.  The reported force
    is the force exerted by the fluid on the cylinder,
    ``p n - mu dev(gradU + gradU.T) n`` integrated per unit depth by default.
    """
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    face_centers = geometry.face_center[boundary_faces]
    cylinder_flag = case.is_cylinder_boundary(face_centers)
    cylinder_faces = boundary_faces[cylinder_flag]
    if cylinder_faces.shape[0] == 0:
        raise ValueError("No cylinder boundary faces were selected.")

    owner = geometry.owner[cylinder_faces]
    sf = geometry.S_f[cylinder_faces]
    pressure_force = pressure[owner, None] * sf

    if viscous_method == "wall_sn_grad":
        viscous_force = _wall_sn_grad_viscous_force(
            geometry, case, cylinder_faces, owner, sf, velocity
        )
    elif viscous_method == "cell_gradient":
        viscous_force = bm.zeros_like(pressure_force)
        if velocity_gradient is not None:
            viscous_force = _cell_gradient_viscous_force(
                case, owner, sf, velocity, velocity_gradient
            )
    elif viscous_method == "none":
        viscous_force = bm.zeros_like(pressure_force)
    else:
        raise ValueError(
            "viscous_method must be 'wall_sn_grad', 'cell_gradient', or 'none'."
        )

    total_force = bm.sum(pressure_force + viscous_force, axis=0)
    pressure_total = bm.sum(pressure_force, axis=0)
    viscous_total = bm.sum(viscous_force, axis=0)

    diameter = 2.0 * float(case.radius)
    dynamic_scale = float(case.rho) * float(case.mean_velocity) ** 2 * diameter
    if dynamic_scale == 0.0:
        drag_coefficient = 0.0
        lift_coefficient = 0.0
        pressure_drag_coefficient = 0.0
        pressure_lift_coefficient = 0.0
        viscous_drag_coefficient = 0.0
        viscous_lift_coefficient = 0.0
    else:
        drag_coefficient = 2.0 * _as_float(total_force[0]) / dynamic_scale
        lift_coefficient = 2.0 * _as_float(total_force[1]) / dynamic_scale
        pressure_drag_coefficient = 2.0 * _as_float(pressure_total[0]) / dynamic_scale
        pressure_lift_coefficient = 2.0 * _as_float(pressure_total[1]) / dynamic_scale
        viscous_drag_coefficient = 2.0 * _as_float(viscous_total[0]) / dynamic_scale
        viscous_lift_coefficient = 2.0 * _as_float(viscous_total[1]) / dynamic_scale

    return {
        "cylinder_faces": int(cylinder_faces.shape[0]),
        "force_x": _as_float(total_force[0]),
        "force_y": _as_float(total_force[1]),
        "pressure_force_x": _as_float(pressure_total[0]),
        "pressure_force_y": _as_float(pressure_total[1]),
        "viscous_force_x": _as_float(viscous_total[0]),
        "viscous_force_y": _as_float(viscous_total[1]),
        "drag_coefficient": float(drag_coefficient),
        "lift_coefficient": float(lift_coefficient),
        "pressure_drag_coefficient": float(pressure_drag_coefficient),
        "pressure_lift_coefficient": float(pressure_lift_coefficient),
        "viscous_drag_coefficient": float(viscous_drag_coefficient),
        "viscous_lift_coefficient": float(viscous_lift_coefficient),
    }


def solution_summary(
    model,
    case,
    solve_result,
    *,
    viscous_method: str = "wall_sn_grad",
    force: dict | None = None,
    probes: dict | None = None,
) -> dict[str, float | int | bool]:
    """Return scalar field diagnostics for a solved cylinder-flow model."""
    velocity = solve_result.velocity
    pressure = solve_result.pressure
    speed = bm.linalg.norm(velocity, axis=1)
    if force is None:
        force = cylinder_force_coefficients(
            model.mesh,
            case,
            velocity=velocity,
            pressure=pressure,
            velocity_gradient=getattr(model, "velocity_gradient", None),
            viscous_method=viscous_method,
            geometry=getattr(model, "fvm_geometry", None),
        )
    if probes is None:
        probes = pressure_drop(model.fvm_geometry.cell_center, pressure)
    return {
        "cells": int(model.fvm_geometry.NC),
        "faces": int(model.fvm_geometry.NF),
        "finite_fields": bool(
            bm.to_numpy(
                bm.all(bm.isfinite(velocity))
                & bm.all(bm.isfinite(pressure))
            )
        ),
        "speed_max": _as_float(bm.max(speed)),
        "speed_mean": _as_float(bm.mean(speed)),
        "u_min": _as_float(bm.min(velocity[:, 0])),
        "u_max": _as_float(bm.max(velocity[:, 0])),
        "v_min": _as_float(bm.min(velocity[:, 1])),
        "v_max": _as_float(bm.max(velocity[:, 1])),
        "pressure_min": _as_float(bm.min(pressure)),
        "pressure_max": _as_float(bm.max(pressure)),
        **{f"force_{key}": value for key, value in force.items()},
        **{f"pressure_drop_{key}": value for key, value in probes.items()},
    }


def strouhal_summary(
    force_history: Iterable[dict],
    *,
    reference_length: float,
    reference_velocity: float,
    start_time: float | None = None,
    min_lift_amplitude: float = 1.0e-3,
    lift_key: str = "lift_coefficient",
    drag_key: str = "drag_coefficient",
) -> dict[str, float | int | bool | None]:
    """Estimate Strouhal data from a lift-coefficient time history."""
    rows = [
        row for row in force_history
        if start_time is None or float(row["time"]) >= float(start_time)
    ]
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    lift = np.asarray([float(row[lift_key]) for row in rows], dtype=float)
    drag = np.asarray([float(row[drag_key]) for row in rows], dtype=float)
    result = {
        "valid": False,
        "samples": int(times.size),
        "peak_count": 0,
        "start_time": None if times.size == 0 else float(times[0]),
        "end_time": None if times.size == 0 else float(times[-1]),
        "mean_drag_coefficient": None if drag.size == 0 else float(np.mean(drag)),
        "mean_lift_coefficient": None if lift.size == 0 else float(np.mean(lift)),
        "lift_amplitude": None,
        "min_lift_amplitude": float(min_lift_amplitude),
        "period_mean": None,
        "period_std": None,
        "frequency": None,
        "strouhal_number": None,
    }
    if times.size < 3 or reference_length <= 0.0 or reference_velocity <= 0.0:
        return result

    result["lift_amplitude"] = 0.5 * float(np.max(lift) - np.min(lift))
    if result["lift_amplitude"] < float(min_lift_amplitude):
        return result

    peak_mask = (lift[1:-1] > lift[:-2]) & (lift[1:-1] >= lift[2:])
    peak_times = times[1:-1][peak_mask]
    result["peak_count"] = int(peak_times.size)
    if peak_times.size < 2:
        return result

    periods = np.diff(peak_times)
    period_mean = float(np.mean(periods))
    if period_mean <= 0.0:
        return result

    frequency = 1.0 / period_mean
    result.update(
        {
            "valid": True,
            "period_mean": period_mean,
            "period_std": float(np.std(periods)),
            "frequency": frequency,
            "strouhal_number": frequency
            * float(reference_length)
            / float(reference_velocity),
        }
    )
    return result


def plot_cylinder_overview(
    model,
    case,
    solve_result,
    output: str | Path,
) -> None:
    """Write a compact PNG overview of speed, pressure, and velocity vectors."""
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = _as_numpy(model.fvm_geometry.cell_center)
    speed = _as_numpy(bm.linalg.norm(solve_result.velocity, axis=1))
    pressure = _as_numpy(solve_result.pressure)
    velocity = _as_numpy(solve_result.velocity)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.0), sharex=True)
    fields = ((speed, "speed", "viridis"), (pressure, "pressure", "coolwarm"))
    for ax, (field, title, cmap) in zip(axes, fields):
        contour = ax.tricontourf(points[:, 0], points[:, 1], field, levels=40, cmap=cmap)
        stride = max(1, points.shape[0] // 250)
        ax.quiver(
            points[::stride, 0],
            points[::stride, 1],
            velocity[::stride, 0],
            velocity[::stride, 1],
            color="k",
            alpha=0.35,
            scale=8.0,
            width=0.0018,
        )
        ax.add_patch(
            plt.Circle(
                case.center,
                case.radius,
                fill=False,
                color="black",
                linewidth=0.8,
            )
        )
        ax.set_xlim(case.box[0], case.box[1])
        ax.set_ylim(case.box[2], case.box[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_cylinder_outputs(
    model,
    case,
    solve_result,
    output_dir: str | Path,
    *,
    velocity_gradient,
    residuals: Iterable[dict] | None = None,
    force_history: Iterable[dict] | None = None,
    strouhal_start_time: float | None = None,
    strouhal_min_lift_amplitude: float = 1.0e-3,
    run_summary: dict | None = None,
    viscous_method: str = "wall_sn_grad",
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure", "speed"),
) -> dict:
    """Write standard cylinder benchmark artifacts for a solved model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_solution_vtk(
        model.mesh,
        solve_result.velocity,
        solve_result.pressure,
        output_dir / "solution.vtu",
        fields=fields,
        velocity_gradient=velocity_gradient,
        geometry=getattr(model, "fvm_geometry", None),
    )
    plot_cylinder_overview(
        model,
        case,
        solve_result,
        output_dir / "flow_overview.png",
    )

    if residuals is not None:
        write_dict_csv(output_dir / "residual_history.csv", scalarize_rows(residuals))
    strouhal = None
    if force_history is not None:
        force_rows = list(force_history)
        write_dict_csv(output_dir / "force_history.csv", scalarize_rows(force_rows))
        strouhal = strouhal_summary(
            force_rows,
            reference_length=2.0 * float(case.radius),
            reference_velocity=abs(float(case.mean_velocity)),
            start_time=strouhal_start_time,
            min_lift_amplitude=strouhal_min_lift_amplitude,
        )
        write_dict_csv(output_dir / "strouhal_summary.csv", [strouhal])

    force = cylinder_force_coefficients(
        model.mesh,
        case,
        velocity=solve_result.velocity,
        pressure=solve_result.pressure,
        velocity_gradient=velocity_gradient,
        viscous_method=viscous_method,
        geometry=getattr(model, "fvm_geometry", None),
    )
    probes = pressure_drop(
        model.fvm_geometry.cell_center,
        solve_result.pressure,
    )
    summary = solution_summary(
        model,
        case,
        solve_result,
        viscous_method=viscous_method,
        force=force,
        probes=probes,
    )
    if residuals is not None:
        residual_rows = list(residuals)
        if residual_rows:
            final_residual = scalarize_rows([residual_rows[-1]])[0]
            summary.update(
                {
                    "iterations": len(residual_rows),
                    "last_mass": final_residual.get(
                        "mass",
                        final_residual.get("mass_relative_l2"),
                    ),
                    "last_pressure_correction": final_residual.get(
                        "pressure_correction",
                        final_residual.get("pressure_correction_l2"),
                    ),
                }
            )
    if run_summary:
        summary.update(run_summary)
    if strouhal is not None:
        summary.update({f"strouhal_{key}": value for key, value in strouhal.items()})

    write_dict_csv(output_dir / "force_summary.csv", [force])
    write_dict_csv(output_dir / "pressure_drop.csv", [probes])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    return {
        "output_dir": output_dir,
        "summary": summary,
        "force": force,
        "pressure_drop": probes,
        "strouhal": strouhal,
    }
