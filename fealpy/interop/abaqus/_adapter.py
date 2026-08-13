"""Adapters from Abaqus external representations to FEALPy domains."""

from __future__ import annotations

from ...backend import bm
from ...mesh.storage import EntitySector, MeshBlock
from ...mesh.view import Mesh
from ._model import AbaqusInpModel


_ABAQUS_LABEL_ATTRIBUTE = "abaqus_label"


class AbaqusMeshAdapterError(ValueError):
    """The selected Abaqus model scope cannot be represented as a mesh."""


def _available_parts(model: AbaqusInpModel) -> set[str | None]:
    return {
        block.part_name
        for block in (*model.node_blocks, *model.element_blocks)
    }


def _select_part(
    model: AbaqusInpModel,
    part_name: str | None,
) -> str | None:
    available = _available_parts(model)
    if not available:
        raise AbaqusMeshAdapterError("the Abaqus model contains no mesh blocks")

    if part_name is not None:
        if part_name not in available:
            names = sorted(name for name in available if name is not None)
            raise AbaqusMeshAdapterError(
                f"Abaqus part {part_name!r} was not found; available parts: {names}"
            )
        return part_name

    if len(available) != 1:
        names = sorted("<global>" if name is None else name for name in available)
        raise AbaqusMeshAdapterError(
            "the Abaqus model contains multiple mesh scopes; select one with "
            f"part_name (available: {names})"
        )
    return next(iter(available))


def _collect_nodes(
    model: AbaqusInpModel,
    part_name: str | None,
) -> tuple[list[int], list[tuple[float, ...]]]:
    labels: list[int] = []
    coordinates: list[tuple[float, ...]] = []
    seen: set[int] = set()

    for block in model.node_blocks:
        if block.part_name != part_name:
            continue
        if len(block.labels) != len(block.coordinates):
            raise AbaqusMeshAdapterError(
                f"node block in part {part_name!r} has inconsistent record counts"
            )
        for label, point in zip(block.labels, block.coordinates):
            if label in seen:
                raise AbaqusMeshAdapterError(
                    f"duplicate node label {label} in part {part_name!r}"
                )
            if len(point) != 3:
                raise AbaqusMeshAdapterError(
                    f"C3D4 node {label} in part {part_name!r} requires three coordinates"
                )
            seen.add(label)
            labels.append(label)
            coordinates.append(point)

    if not labels:
        raise AbaqusMeshAdapterError(
            f"Abaqus part {part_name!r} contains no nodes"
        )
    return labels, coordinates


def _collect_elements(
    model: AbaqusInpModel,
    part_name: str | None,
    node_index: dict[int, int],
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    labels: list[int] = []
    connectivity: list[tuple[int, int, int, int]] = []
    seen: set[int] = set()

    for block in model.element_blocks:
        if block.part_name != part_name:
            continue
        if block.element_type.upper() != "C3D4":
            raise AbaqusMeshAdapterError(
                f"unsupported Abaqus element type {block.element_type!r}; "
                "current mesh adapter is limited to C3D4"
            )
        if len(block.labels) != len(block.connectivity):
            raise AbaqusMeshAdapterError(
                f"element block in part {part_name!r} has inconsistent record counts"
            )
        for label, abaqus_nodes in zip(block.labels, block.connectivity):
            if label in seen:
                raise AbaqusMeshAdapterError(
                    f"duplicate element label {label} in part {part_name!r}"
                )
            if len(abaqus_nodes) != 4:
                raise AbaqusMeshAdapterError(
                    f"C3D4 element {label} in part {part_name!r} requires four nodes"
                )
            missing = sorted({node for node in abaqus_nodes if node not in node_index})
            if missing:
                raise AbaqusMeshAdapterError(
                    f"element {label} in part {part_name!r} references undefined "
                    f"node labels {missing}"
                )
            seen.add(label)
            labels.append(label)
            connectivity.append(tuple(node_index[node] for node in abaqus_nodes))

    if not labels:
        raise AbaqusMeshAdapterError(
            f"Abaqus part {part_name!r} contains no C3D4 elements"
        )
    return labels, connectivity


def abaqus_model_to_mesh(
    model: AbaqusInpModel,
    *,
    part_name: str | None = None,
) -> Mesh:
    """Convert one Abaqus C3D4 part into a FEALPy mesh.

    Multiple ``*Node`` and ``*Element`` blocks in the selected part are merged
    in source order.  Abaqus labels are mapped to contiguous FEALPy indices and
    retained as ``abaqus_label`` attributes on the node and tetrahedron entity
    sectors.  No lower-dimensional topology is constructed here.

    Assembly instances are outside the current support coverage.  A model with
    multiple parts therefore requires an explicit ``part_name`` selection.
    """

    selected_part = _select_part(model, part_name)
    node_labels, coordinates = _collect_nodes(model, selected_part)
    node_index = {label: index for index, label in enumerate(node_labels)}
    element_labels, connectivity = _collect_elements(
        model,
        selected_part,
        node_index,
    )

    positions = bm.asarray(coordinates, dtype=bm.float64)
    block = MeshBlock(positions=positions)

    node_indices = bm.reshape(
        bm.arange(len(node_labels), dtype=bm.int64),
        (-1, 1),
    )
    node_sector = EntitySector("node", node_indices)
    node_sector.attributes[_ABAQUS_LABEL_ATTRIBUTE] = bm.asarray(
        node_labels,
        dtype=bm.int64,
    )
    block.add_sector(node_sector)

    tet_sector = EntitySector(
        "tet",
        bm.asarray(connectivity, dtype=bm.int64),
    )
    tet_sector.attributes[_ABAQUS_LABEL_ATTRIBUTE] = bm.asarray(
        element_labels,
        dtype=bm.int64,
    )
    block.add_sector(tet_sector, root=True)

    return Mesh(block)
