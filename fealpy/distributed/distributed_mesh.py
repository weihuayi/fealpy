from __future__ import annotations

__all__ = [
    "distribute_mesh",
    "DistMeshResult",
    "MeshComm",
]

from typing import Any, NamedTuple
from collections import defaultdict
from collections.abc import Sequence, Mapping

from mpi4py.MPI import Comm, COMM_WORLD

from ..backend import bm, Tensor
from ..mesh import Mesh, MeshBlock, EntitySector, Relation
from . import entity_mpi as _de


class MeshComm(NamedTuple):
    entities: dict[str, _de.EntityMPI]
    root_entity_name: str


class DistMeshResult(NamedTuple):
    mesh: Mesh
    comm: MeshComm


def _build_masks_by_sector(
    storage: MeshBlock,
    root_name: str,
    root_masks: Sequence[Tensor],
) -> dict[str, list[Tensor]]:
    num_parts = len(root_masks)
    block_dims = {name: storage.get_sector(name).schema.top_dim for name in storage.sectors}
    root_dim = block_dims[root_name]

    masks_by_sector: dict[str, list[Tensor]] = {
        name: [
            bm.full((storage.get_sector(name).indices.shape[0],), False, dtype=bm.bool)
            for _ in range(num_parts)
        ]
        for name in storage.sectors
    }
    masks_by_sector[root_name] = [bm.asarray(mask) for mask in root_masks]

    outgoing_relations: dict[str, list[tuple[str, Relation]]] = defaultdict(list)
    for (src_name, tgt_name), relation in storage.relations.items():
        if relation.src_indices is not None:
            continue

        if block_dims[src_name] <= block_dims[tgt_name]:
            continue

        outgoing_relations[src_name].append((tgt_name, relation))

    dims_desc = sorted({dim for dim in block_dims.values() if dim <= root_dim}, reverse=True)
    for pid in range(num_parts):
        for dim in dims_desc:
            for src_name, src_dim in block_dims.items():
                if src_dim != dim:
                    continue

                src_mask = masks_by_sector[src_name][pid]
                if not bool(bm.any(src_mask)):
                    continue

                for tgt_name, relation in outgoing_relations.get(src_name, []):
                    selected_tgt = bm.reshape(bm.asarray(relation.tgt_indices[src_mask]), (-1,))
                    if int(selected_tgt.shape[0]) == 0:
                        continue

                    masks_by_sector[tgt_name][pid][selected_tgt] = True  # type: ignore[index]

    return masks_by_sector


def _build_local_storage(
    storage: MeshBlock,
    masks_by_sector: Mapping[str, Sequence[Tensor]],
    part_id: int,
) -> MeshBlock:
    root_names = list(storage.root_entity_names)
    if len(root_names) != 1:
        raise ValueError("only one root entity is supported when building local storage")

    block_dims = {name: storage.get_sector(name).schema.top_dim for name in storage.sectors}
    block_maps: dict[str, Tensor] = {}
    selected_indices_by_block: dict[str, Tensor] = {}

    for name, block in storage.sectors.items():
        block_mask = bm.asarray(masks_by_sector[name][part_id])
        block_map = _make_local_index(block.indices.shape[0], block_mask, nonlocal_value=-1)
        block_maps[name] = block_map
        selected_indices_by_block[name] = bm.asarray(block.indices[block_mask])

    local_relations: dict[tuple[str, str], Relation] = {}
    for key, relation in storage.relations.items():
        src_name, tgt_name = key
        if relation.src_indices is not None:
            continue

        if block_dims[src_name] <= block_dims[tgt_name]:
            continue

        src_mask = bm.asarray(masks_by_sector[src_name][part_id])
        local_tgt_indices = bm.asarray(block_maps[tgt_name][relation.tgt_indices[src_mask]])
        if bool(bm.any(local_tgt_indices < 0)):  # type: ignore[arg-type]
            raise ValueError(
                f"partition {part_id}: found non-local targets in relation {key!r}"
            )

        local_relations[key] = Relation(
            src_name=src_name,
            tgt_name=tgt_name,
            tgt_indices=local_tgt_indices,
            src_indices=None,
        )

    position_mask = bm.full((len(storage.positions),), False, dtype=bm.bool)
    for indices in selected_indices_by_block.values():
        flat = bm.reshape(bm.asarray(indices), (-1,))
        if int(flat.shape[0]) == 0:
            continue
        position_mask[flat] = True  # type: ignore[index]

    position_map = _make_local_index(len(storage.positions), position_mask, nonlocal_value=-1)
    lpos = bm.asarray(storage.positions[position_mask])
    local_storage = MeshBlock(positions=lpos, root_entity_names=root_names)

    for name, block in storage.sectors.items():
        local_indices = bm.asarray(position_map[selected_indices_by_block[name]])
        if bool(bm.any(local_indices < 0)):  # type: ignore[arg-type]
            raise ValueError(
                f"partition {part_id}: found non-local positions in block {name!r}"
            )

        local_storage.add_sector(
            EntitySector(
                schema_name=name,
                indices=local_indices,
                attributes=dict(block.attributes),
            ),
            root=(name in root_names),
        )

    local_storage.relations.update(local_relations)

    return local_storage


def distribute_mesh(
    mesh: Mesh | None,
    cell_masks: Sequence[Tensor] | None,
    *,
    root: int = 0,
    comm: Comm | None = None,
) -> DistMeshResult:
    """Split a shape-block mesh into multiple parts based on MPI.

    Parameters:
        mesh (Mesh | None): The mesh to be split, only required in root.
        cell_masks (Sequence[Tensor] | None): Root-entity mask tensors for partitions,
            only required in root.
        root (int, optional): The root rank. Defaults to 0.
        comm (Comm, optional): The MPI communicator to use. Defaults to None.

    Returns:
        namedtuple:
        - mesh (Mesh): Mesh partition in this process.
        - comm (namedtuple):
                    entities: Entity exchangers for each block name in storage.
                    root_entity_name: The root entity key in the storage.
    """
    if comm is None:
        comm = COMM_WORLD

    local_storage_list: list[MeshBlock] | None = None
    gdata: dict[str, Any] = {}

    if comm.Get_rank() == root:
        assert mesh is not None, "root: Mesh must be provided when root."
        assert cell_masks is not None, "root: Cell masks must be provided when root."

        if len(cell_masks) != comm.Get_size():
            raise ValueError("root: Number of cell masks must equal comm size.")

        root_names = list(mesh.block.root_entity_names)
        if len(root_names) != 1:
            raise ValueError("root: Exactly one root entity is required to distribute mesh.")

        root_name = root_names[0]
        expected_num_cells = mesh.block.get_sector(root_name).indices.shape[0]

        for cell_mask in cell_masks:
            if not bm.any(cell_mask):
                raise ValueError("root: All cell masks must be non-empty.")
            if int(cell_mask.shape[0]) != int(expected_num_cells):
                raise ValueError("root: Cell mask shape does not match root entity count.")

        masks_by_sector = _build_masks_by_sector(mesh.block, root_name, cell_masks)
        local_storage_list = [
            _build_local_storage(mesh.block, masks_by_sector, pid)
            for pid in range(comm.Get_size())
        ]

        gdata = {
            "root_name": root_name,
            "masks_by_sector": masks_by_sector,
        }

    gdata = comm.bcast(gdata, root)
    lstorage = comm.scatter(local_storage_list, root)
    pmesh = Mesh(lstorage)

    entities_comm: dict[str, _de.EntityMPI] = {}
    for name, masks in gdata["masks_by_sector"].items():
        entities_comm[name] = _de.dist_from_masks(masks, comm=comm)

    return DistMeshResult(
        pmesh.fealpy_api(),
        MeshComm(entities_comm, gdata["root_name"]),
    )


def _make_local_index(num: int, mask: Tensor, nonlocal_value: int = 0):
    local_index = bm.full((num,), nonlocal_value, dtype=bm.int32)
    lnum = int(bm.sum(mask))  # type: ignore[arg-type]
    local_index[mask] = bm.arange(lnum, dtype=bm.int32)  # type: ignore[index]
    return local_index
