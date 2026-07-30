"""Common output helpers for FVM benchmark examples."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import write_mesh_to_vtu
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


def re_label(re: float) -> str:
    """Return a filesystem-stable Reynolds-number label."""
    text = f"{float(re):g}".replace(".", "p").replace("-", "m")
    return f"Re{text}"


def scalarize_value(value):
    """Convert backend scalar or array values to CSV-friendly objects."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    array = np.asarray(bm.to_numpy(value))
    if array.shape == ():
        return array.item()
    return array.tolist()


def scalarize_rows(rows: Iterable[dict]) -> list[dict]:
    """Convert backend scalar values in dictionaries to CSV-friendly objects."""
    result = []
    for row in rows:
        values = asdict(row) if is_dataclass(row) else row
        result.append(
            {
                key: scalarize_value(value)
                for key, value in values.items()
            }
        )
    return result


def write_dict_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write a sequence of dictionaries as CSV."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solution_cell_fields(
    velocity: TensorLike,
    pressure: TensorLike,
    *,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure"),
    velocity_gradient=None,
) -> dict[str, TensorLike]:
    """Build selected cell fields for VTU output."""
    if velocity.ndim != 2 or velocity.shape[1] < 2:
        raise ValueError("velocity must have shape (NC, GD) with GD >= 2.")
    u = velocity[:, 0]
    v = velocity[:, 1]
    available = {
        "velocity": velocity,
        "u": u,
        "v": v,
        "pressure": pressure,
        "speed": bm.linalg.norm(velocity, axis=1),
    }
    if "vorticity" in fields:
        if velocity_gradient is None:
            raise ValueError("velocity_gradient is required for vorticity output.")
        grad = velocity_gradient.cell_gradient(velocity)
        available["vorticity"] = grad[:, 1, 0] - grad[:, 0, 1]
    return {name: available[name] for name in fields}


def write_solution_vtk(
    mesh,
    velocity: TensorLike,
    pressure: TensorLike,
    path: str | Path,
    *,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure"),
    velocity_gradient=None,
    geometry: FVMGeometry | None = None,
) -> None:
    """Attach global FVM cell fields to one root sector and write a VTU file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    if len(geometry.cell_views) != 1:
        raise RuntimeError(
            "mixed cell-sector VTU output is unavailable until "
            "fealpy.mesh.write_mesh_to_vtu merges same-named cell fields "
            "across sectors."
        )
    cell_fields = solution_cell_fields(
        velocity,
        pressure,
        fields=fields,
        velocity_gradient=velocity_gradient,
    )
    for view, cell_slice in zip(
        geometry.cell_views, geometry.cell_sector_slices
    ):
        for name, value in cell_fields.items():
            view.set_attribute(name, value[cell_slice])
    write_mesh_to_vtu(
        str(path),
        mesh,
        entity_names=[view.sector.schema_name for view in geometry.cell_views],
    )
