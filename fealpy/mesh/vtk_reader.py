from __future__ import annotations

from collections.abc import Sequence
import importlib
from typing import Any

import numpy as np

from ..backend import bm, Tensor
from .schema import SCHEMA_REGISTRY
from .storage import EntitySector, MeshBlock
from .topology.builder import TopologyBuilder
from .view import Mesh
from .vtk_writter import SCHEMA_TO_VTK_CELL_TYPE_NAME

__all__ = [
    "VTK_CELL_TYPE_TO_SCHEMA",
    "read_mesh_from_vtu",
]


def _load_vtk():
    try:
        vtk = importlib.import_module("vtk")
        vnp = importlib.import_module("vtk.util.numpy_support")
    except ImportError as exc:
        raise ImportError(
            "Reading VTU files requires the 'vtk' package. "
            "Please install it first (e.g. `pip install vtk`)."
        ) from exc
    return vtk, vnp


def _build_vtk_cell_type_to_schema(vtk_mod) -> dict[int, str]:
    return {
        int(getattr(vtk_mod, vtk_name)): schema_name
        for schema_name, vtk_name in SCHEMA_TO_VTK_CELL_TYPE_NAME.items()
    }


def _vtk_array_to_numpy(vtk_array: Any, vnp_mod) -> np.ndarray:
    arr = vnp_mod.vtk_to_numpy(vtk_array)
    return np.asarray(arr)


def _array_name(vtk_array: Any) -> str | None:
    name = vtk_array.GetName()
    return None if name in (None, "") else str(name)


def _point_ids(cell: Any) -> list[int]:
    ids = cell.GetPointIds()
    return [int(ids.GetId(i)) for i in range(ids.GetNumberOfIds())]


def _infer_geometric_dimension(points: np.ndarray) -> int:
    if points.ndim != 2:
        raise ValueError(f"Expected VTU points with shape (N, GD), got shape {points.shape}.")

    gd = points.shape[1]
    while gd > 1 and np.allclose(points[:, gd - 1], 0.0):
        gd -= 1
    return gd


def _normalize_geometric_dimension(
    points: np.ndarray,
    geometric_dimension: int | None,
) -> Tensor:
    if geometric_dimension is None:
        geometric_dimension = _infer_geometric_dimension(points)
    if geometric_dimension < 1 or geometric_dimension > points.shape[1]:
        raise ValueError(
            f"geometric_dimension must be between 1 and {points.shape[1]}, "
            f"got {geometric_dimension}."
        )
    return bm.asarray(points[:, :geometric_dimension])


def _iter_data_arrays(attributes: Any, vnp_mod) -> list[tuple[str, np.ndarray]]:
    arrays: list[tuple[str, np.ndarray]] = []
    for i in range(attributes.GetNumberOfArrays()):
        vtk_array = attributes.GetArray(i)
        name = _array_name(vtk_array)
        if name is None:
            continue
        arrays.append((name, _vtk_array_to_numpy(vtk_array, vnp_mod)))
    return arrays


def _root_entity_names(schema_names: Sequence[str]) -> list[str]:
    present = set(schema_names)
    roots: list[str] = []
    for schema_name in present:
        schema = SCHEMA_REGISTRY[schema_name]
        if not any(
            other != schema_name and SCHEMA_REGISTRY[other].top_dim > schema.top_dim
            for other in present
        ):
            roots.append(schema_name)
    return sorted(roots, key=lambda name: (-SCHEMA_REGISTRY[name].top_dim, name))


def _cell_blocks_from_arrays(
    grid: Any,
    cell_type_to_schema: dict[int, str],
    vnp_mod,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    types = _vtk_array_to_numpy(grid.GetCellTypes(), vnp_mod).astype(np.int64, copy=False)
    cells = grid.GetCells()
    connectivity = _vtk_array_to_numpy(cells.GetConnectivityArray(), vnp_mod).astype(np.int64, copy=False)
    offsets = _vtk_array_to_numpy(cells.GetOffsetsArray(), vnp_mod).astype(np.int64, copy=False)

    total_cells = int(grid.GetNumberOfCells())
    if types.shape[0] != total_cells:
        raise ValueError(
            f"Cell type array has incompatible length: expected {total_cells}, got {types.shape[0]}."
        )
    if offsets.shape[0] != total_cells + 1:
        raise ValueError(
            f"Cell offsets array has incompatible length: expected {total_cells + 1}, got {offsets.shape[0]}."
        )
    if offsets[0] != 0 or offsets[-1] != connectivity.shape[0]:
        raise ValueError("Cell offsets are inconsistent with connectivity length.")

    blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen_types: list[int] = []
    for cell_type in types:
        cell_type_int = int(cell_type)
        if cell_type_int not in seen_types:
            seen_types.append(cell_type_int)

    for cell_type in seen_types:
        try:
            schema_name = cell_type_to_schema[cell_type]
        except KeyError as exc:
            first = int(np.flatnonzero(types == cell_type)[0])
            raise ValueError(f"Unsupported VTK cell type {cell_type} at cell {first}.") from exc

        cell_indices = np.flatnonzero(types == cell_type).astype(np.int64, copy=False)
        widths = offsets[cell_indices + 1] - offsets[cell_indices]
        if widths.shape[0] == 0:
            continue
        width = int(widths[0])
        if not np.all(widths == width):
            raise ValueError(f"VTK cell type {cell_type} has variable connectivity width.")

        if np.all(cell_indices[1:] == cell_indices[:-1] + 1):
            start = int(offsets[cell_indices[0]])
            end = int(offsets[cell_indices[-1] + 1])
            indices = connectivity[start:end].reshape((-1, width))
        else:
            starts = offsets[cell_indices]
            indices = connectivity[starts[:, None] + np.arange(width, dtype=np.int64)]
        blocks.append((schema_name, indices, cell_indices))

    return blocks


def _cell_blocks_from_get_cell(
    grid: Any,
    cell_type_to_schema: dict[int, str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    grouped_cells: dict[str, list[list[int]]] = {}
    grouped_indices: dict[str, list[int]] = {}
    for cell_index in range(grid.GetNumberOfCells()):
        cell_type = int(grid.GetCellType(cell_index))
        try:
            schema_name = cell_type_to_schema[cell_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported VTK cell type {cell_type} at cell {cell_index}.") from exc
        grouped_cells.setdefault(schema_name, []).append(_point_ids(grid.GetCell(cell_index)))
        grouped_indices.setdefault(schema_name, []).append(cell_index)

    return [
        (
            schema_name,
            np.asarray(cells, dtype=np.int64),
            np.asarray(grouped_indices[schema_name], dtype=np.int64),
        )
        for schema_name, cells in grouped_cells.items()
    ]


def _cell_blocks(
    grid: Any,
    cell_type_to_schema: dict[int, str],
    vnp_mod,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    try:
        return _cell_blocks_from_arrays(grid, cell_type_to_schema, vnp_mod)
    except AttributeError:
        return _cell_blocks_from_get_cell(grid, cell_type_to_schema)


VTK_CELL_TYPE_TO_SCHEMA: dict[int, str] = {}


def read_mesh_from_vtu(
    filename: str,
    *,
    geometric_dimension: int | None = None,
    construct_topology: bool = True,
) -> Mesh:
    """Read a VTU file and rebuild a :class:`Mesh`.

    Parameters:
        filename (str): Input `.vtu` file path.
        geometric_dimension (int | None, optional): Geometric dimension of the
            returned point array. If omitted, trailing all-zero coordinate columns
            introduced by VTU's 3D point storage are removed.
        construct_topology (bool, optional): If ``True``, construct lower-
            dimensional topology relations from root entities after reading.
    """
    vtk, vnp = _load_vtk()
    cell_type_to_schema = _build_vtk_cell_type_to_schema(vtk)
    VTK_CELL_TYPE_TO_SCHEMA.clear()
    VTK_CELL_TYPE_TO_SCHEMA.update(cell_type_to_schema)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()

    if grid.GetPoints() is None:
        raise ValueError(f"VTU file {filename!r} does not contain points.")

    points = _vtk_array_to_numpy(grid.GetPoints().GetData(), vnp)
    positions = _normalize_geometric_dimension(points, geometric_dimension)
    block = MeshBlock(positions=positions)

    cell_blocks = _cell_blocks(grid, cell_type_to_schema, vnp)
    total_cells = int(grid.GetNumberOfCells())

    root_names = _root_entity_names([schema_name for schema_name, _, _ in cell_blocks])
    cell_indices_by_schema: dict[str, np.ndarray] = {}
    for schema_name, indices, cell_indices in cell_blocks:
        block.add_sector(
            EntitySector(schema_name, bm.asarray(indices, dtype=np.int64)),
            root=schema_name in root_names,
        )
        cell_indices_by_schema[schema_name] = cell_indices

    if not block.has_sector("point"):
        point_indices = bm.arange(positions.shape[0], dtype=np.int64).reshape((-1, 1))
        block.add_sector(EntitySector("point", point_indices), root=not root_names)

    for name, values in _iter_data_arrays(grid.GetPointData(), vnp):
        if values.shape[0] != positions.shape[0]:
            raise ValueError(
                f"Point attributes {name!r} has incompatible leading dimension: "
                f"expected {positions.shape[0]}, got {values.shape[0]}."
            )
        block.get_sector("point").attributes[name] = values

    cell_data_arrays = _iter_data_arrays(grid.GetCellData(), vnp)
    for schema_name, cell_indices in cell_indices_by_schema.items():
        sector = block.get_sector(schema_name)
        for name, values in cell_data_arrays:
            if values.shape[0] != total_cells:
                raise ValueError(
                    f"Cell attributes {name!r} has incompatible leading dimension: "
                    f"expected {total_cells}, got {values.shape[0]}."
                )
            sector.attributes[name] = values[cell_indices]

    if construct_topology and block.root_entity_names:
        root_dims = [SCHEMA_REGISTRY[name].top_dim for name in block.root_entity_names]
        min_root_dim = min(root_dims)
        existing_lower = [
            name for name in block.sectors
            if name not in block.root_entity_names and SCHEMA_REGISTRY[name].top_dim < min_root_dim
        ]
        TopologyBuilder.construct(block, exclude=existing_lower)

    return Mesh(block)
