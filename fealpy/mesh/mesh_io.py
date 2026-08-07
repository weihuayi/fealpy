
"""Mesh input/output helpers backed by :mod:`meshio`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..backend import bm
from .storage import EntitySector, MeshBlock
from .topology.builder import TopologyBuilder

__all__ = ["read", "write"]


_MESHIO_TO_FEALPY = {
    "vertex": "point",
    "line": "segment",
    "triangle": "tri",
    "quad": "quad",
    "tetra": "tet",
    "hexahedron": "hex",
    "wedge": "prism",
    "pyramid": "pyramid",
}
_FEALPY_TO_MESHIO = {v: k for k, v in _MESHIO_TO_FEALPY.items()}


def read(filename: str | Path, file_format: str | None) -> MeshBlock:
    """Read a mesh file and return its :class:`MeshBlock`.

    All supported cell blocks are retained.  Blocks with the same FEALPy
    schema are concatenated into one sector.
    """
    import meshio

    data = meshio.read(filename, file_format=file_format)
    cells = [(block.type, np.asarray(block.data)) for block in data.cells]
    cells = [(name, values) for name, values in cells if name in _MESHIO_TO_FEALPY]
    if not cells:
        raise ValueError(f"No supported cell types found in {filename!s}")
    grouped: dict[str, list[np.ndarray]] = {}
    for cell_type, values in cells:
        schema = _MESHIO_TO_FEALPY[cell_type]
        if schema == "point":
            raise NotImplementedError("Point meshes are not supported by mesh IO yet")
        grouped.setdefault(schema, []).append(values)

    block = MeshBlock(positions=bm.asarray(np.asarray(data.points)))
    for schema, values in grouped.items():
        cell = np.concatenate(values, axis=0)
        block.add_sector(
            EntitySector(schema, bm.asarray(cell, dtype=bm.int32)),
            root=True,
        )
    TopologyBuilder.construct(block)
    return block


def write(
    filename: str | Path,
    block: MeshBlock,
    entity_names: list[str],
    file_format: str | None = None,
    **kwargs: Any
) -> None:
    """Write selected entity sectors from a :class:`MeshBlock`.

    ``entity_names`` contains concrete schema names such as `["tri", "quad"]`;
    aliases like ``"cell"`` are intentionally not accepted.
    """
    import meshio

    if not entity_names:
        raise ValueError("entity_names must contain at least one entity name")
    cells = []
    for schema in entity_names:
        if schema not in block.sectors:
            raise KeyError(f"entity sector {schema!r} is not present in block")
        try:
            cell_type = _FEALPY_TO_MESHIO[schema]
        except KeyError as exc:
            raise ValueError(f"Unsupported FEALPy cell type: {schema!r}") from exc
        cells.append((cell_type, bm.to_numpy(block.sectors[schema].indices)))

    points = bm.to_numpy(block.positions)
    meshio.write(
        filename, meshio.Mesh(points=points, cells=cells),
        file_format=file_format, **kwargs
    )
