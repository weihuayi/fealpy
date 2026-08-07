"""Post-processing helpers for the 2D lid-driven cavity benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .benchmark_postprocess import (
    re_label,
    scalarize_value,
    scalarize_rows,
    write_dict_csv,
    write_solution_vtk,
)
from .fvm_geometry import FVMGeometry
from .gradient_reconstruct import GradientReconstruct
from .piso_result import PisoSnapshot


def _as_numpy(values: TensorLike) -> np.ndarray:
    return np.asarray(bm.to_numpy(values))


def _as_backend(values: np.ndarray, reference: TensorLike) -> TensorLike:
    return bm.array(
        values,
        dtype=reference.dtype,
        device=bm.get_device(reference),
    )


def sample_centerline(
    points: TensorLike,
    values: TensorLike,
    *,
    fixed_axis: int,
    fixed_value: float,
    coordinate_axis: int,
    atol: float | None = None,
) -> TensorLike:
    """Sample scalar cell values along the nearest available centerline."""
    point_array = _as_numpy(points)
    value_array = _as_numpy(values)
    distance = np.abs(point_array[:, fixed_axis] - fixed_value)
    if atol is None:
        mask = distance <= distance.min() + 1.0e-12
    else:
        mask = distance <= atol
        if not np.any(mask):
            mask = distance <= distance.min() + 1.0e-12

    coordinates = point_array[mask, coordinate_axis]
    selected_values = value_array[mask]
    unique_coordinates, inverse = np.unique(coordinates, return_inverse=True)
    averaged_values = np.array(
        [selected_values[inverse == i].mean() for i in range(unique_coordinates.size)]
    )
    profile = np.column_stack([unique_coordinates, averaged_values])
    return _as_backend(profile, points)


def centerline_velocity_profiles(
    points: TensorLike,
    velocity: TensorLike,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    atol: float | None = None,
) -> tuple[TensorLike, TensorLike]:
    """Return ``u(x=center_x, y)`` and ``v(x, y=center_y)`` profiles."""
    u_profile = sample_centerline(
        points,
        velocity[:, 0],
        fixed_axis=0,
        fixed_value=center[0],
        coordinate_axis=1,
        atol=atol,
    )
    v_profile = sample_centerline(
        points,
        velocity[:, 1],
        fixed_axis=1,
        fixed_value=center[1],
        coordinate_axis=0,
        atol=atol,
    )
    return u_profile, v_profile


def primary_vortex_summary(
    points: TensorLike,
    velocity: TensorLike,
    *,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    boundary_margin: float = 0.0,
) -> dict[str, float | int]:
    """Approximate the primary vortex center by the minimum interior speed."""
    point_array = _as_numpy(points)
    velocity_array = _as_numpy(velocity)
    xmin, xmax, ymin, ymax = domain
    interior = (
        (point_array[:, 0] >= xmin + boundary_margin)
        & (point_array[:, 0] <= xmax - boundary_margin)
        & (point_array[:, 1] >= ymin + boundary_margin)
        & (point_array[:, 1] <= ymax - boundary_margin)
    )
    if not np.any(interior):
        interior = np.ones(point_array.shape[0], dtype=bool)

    speed = np.linalg.norm(velocity_array, axis=1)
    candidates = np.nonzero(interior)[0]
    cell_index = int(candidates[np.argmin(speed[candidates])])
    return {
        "cell_index": cell_index,
        "x": float(point_array[cell_index, 0]),
        "y": float(point_array[cell_index, 1]),
        "u": float(velocity_array[cell_index, 0]),
        "v": float(velocity_array[cell_index, 1]),
        "speed": float(speed[cell_index]),
    }


def write_profile_csv(
    path: str | Path,
    profile: TensorLike,
    coordinate_name: str,
    value_name: str,
) -> None:
    """Write a two-column centerline profile CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile_array = _as_numpy(profile)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([coordinate_name, value_name])
        for coordinate, value in profile_array:
            writer.writerow([f"{coordinate:.16g}", f"{value:.16g}"])


@dataclass(frozen=True)
class CavityOutputConfig:
    output_dir: Path
    write_vtk: bool = True
    write_final: bool = True
    write_interval_steps: int | None = None
    write_interval_time: float | None = None
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure", "speed")


def mesh_label(mesh_type: str, nx: int, ny: int) -> str:
    """Return a compact mesh label for output directories."""
    prefix = mesh_type
    if prefix.startswith("uniform_"):
        prefix = prefix[len("uniform_") :]
    return f"{prefix}_{int(nx)}x{int(ny)}"


def default_output_dir(
    solver: str,
    re: float,
    mesh_type: str,
    nx: int,
    ny: int,
    *,
    root: str | Path = "output/lid_driven_cavity",
) -> Path:
    """Return a stable output directory for one cavity benchmark run."""
    return Path(root) / solver / re_label(re) / mesh_label(mesh_type, nx, ny)


def should_write_snapshot(
    step: int,
    time: float,
    config: CavityOutputConfig,
    *,
    is_final: bool = False,
) -> bool:
    """Return whether a snapshot should be written for this time step."""
    if is_final:
        return bool(config.write_final)
    if config.write_interval_steps is not None and config.write_interval_steps > 0:
        if step % config.write_interval_steps == 0:
            return True
    if config.write_interval_time is not None and config.write_interval_time > 0.0:
        quotient = time / config.write_interval_time
        if abs(quotient - round(quotient)) <= 1.0e-12:
            return True
    return False


class CavitySnapshotWriter:
    """Write selected transient cavity snapshots and collect time history."""

    def __init__(
        self,
        config: CavityOutputConfig,
        *,
        geometry: FVMGeometry,
        velocity_gradient: GradientReconstruct,
        domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        total_steps: int | None = None,
        boundary_margin: float = 0.05,
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.velocity_gradient = velocity_gradient
        self.domain = domain
        self.total_steps = total_steps
        self.boundary_margin = boundary_margin
        self.history: list[dict] = []
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
        speed = bm.linalg.norm(cell_velocity, axis=1)
        vortex = primary_vortex_summary(
            self.geometry.cell_center,
            cell_velocity,
            domain=self.domain,
            boundary_margin=self.boundary_margin,
        )
        row = {
            "step": int(step),
            "time": float(time),
            "max_speed": scalarize_value(bm.max(speed)),
            "velocity_update": self._velocity_update(cell_velocity),
            "mass_residual": self._mass_residual(flux),
            "vortex_x": vortex["x"],
            "vortex_y": vortex["y"],
        }
        self.history.append(row)

        is_final = (
            self.total_steps is not None
            and int(step) == int(self.total_steps)
        )
        if self.config.write_vtk and should_write_snapshot(
            step, time, self.config, is_final=is_final
        ):
            self.write_snapshot(
                step,
                cell_velocity,
                pressure,
            )

        self._previous_velocity = bm.copy(cell_velocity)

    def _velocity_update(self, cell_velocity):
        if self._previous_velocity is None:
            return 0.0
        delta = cell_velocity - self._previous_velocity
        return scalarize_value(bm.max(bm.abs(delta)))

    def _mass_residual(self, flux):
        if flux is None:
            return None
        imbalance = self.geometry.scatter_face_flux_to_cells(flux)
        return scalarize_value(bm.max(bm.abs(imbalance)))

    def write_snapshot(
        self,
        step: int,
        cell_velocity,
        pressure,
    ) -> None:
        snapshot_dir = self.config.output_dir / f"{int(step):06d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_solution_vtk(
            self.geometry.mesh,
            cell_velocity,
            pressure,
            snapshot_dir / "solution.vtu",
            fields=self.config.fields,
            velocity_gradient=self.velocity_gradient,
            geometry=self.geometry,
        )

    def write_time_history(self) -> None:
        write_dict_csv(self.config.output_dir / "time_history.csv", self.history)


def write_benchmark_outputs(
    model,
    output_dir: str | Path,
    *,
    velocity: TensorLike,
    pressure: TensorLike,
    residuals: Iterable[dict] | None = None,
    velocity_gradient: TensorLike | None = None,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    boundary_margin: float = 0.05,
    run_summary: dict | None = None,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure", "speed"),
) -> dict:
    """Write standard cavity benchmark outputs for a solved model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = model.fvm_geometry.cell_center
    u_profile, v_profile = centerline_velocity_profiles(points, velocity)
    vortex = primary_vortex_summary(
        points,
        velocity,
        domain=domain,
        boundary_margin=boundary_margin,
    )

    write_profile_csv(output_dir / "centerline_u.csv", u_profile, "y", "u")
    write_profile_csv(output_dir / "centerline_v.csv", v_profile, "x", "v")
    write_dict_csv(output_dir / "vortex_summary.csv", [vortex])
    write_solution_vtk(
        model.mesh,
        velocity,
        pressure,
        output_dir / "solution.vtu",
        fields=fields,
        velocity_gradient=velocity_gradient,
    )

    if residuals is not None:
        write_dict_csv(output_dir / "residual_history.csv", scalarize_rows(residuals))

    if run_summary is not None:
        lines = [f"{key}: {value}" for key, value in run_summary.items()]
        (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")

    return {
        "output_dir": output_dir,
        "u_profile": u_profile,
        "v_profile": v_profile,
        "vortex": vortex,
    }
