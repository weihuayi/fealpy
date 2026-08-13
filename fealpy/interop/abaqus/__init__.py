"""Abaqus interoperability support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...mesh.view import Mesh


__all__ = ["read_inp"]


def read_inp(
    filename: str | Path,
    *,
    part_name: str | None = None,
) -> Mesh:
    """Read one Abaqus C3D4 part as a FEALPy mesh.

    The returned mesh contains node and tetrahedron sectors but does not build
    lower-dimensional topology.  Call ``mesh.construct(...)`` explicitly when
    faces or other derived entities are needed, for example before plotting a
    three-dimensional mesh.

    Parameters:
        filename: Abaqus INP file to read.
        part_name: Part to convert when the file contains multiple mesh scopes.

    Returns:
        A FEALPy mesh containing the selected C3D4 part.
    """

    from ._adapter import abaqus_model_to_mesh
    from ._codec import read_inp_model

    return abaqus_model_to_mesh(
        read_inp_model(filename),
        part_name=part_name,
    )
